#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fahren/fahren.h>

extern "C" {
#include "../../src/internal.h"
}

static int g_cuda_ok = -1;

int fahren_cuda_init(void) {
    if (g_cuda_ok >= 0) return g_cuda_ok;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        g_cuda_ok = 0;
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        g_cuda_ok = 0;
        return 0;
    }
    g_cuda_ok = 1;
    return 1;
}

__global__ void add_bias_kernel(float* y, const float* b, int out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out) y[o] += b[o];
}

__global__ void relu_kernel(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && y[i] < 0.0f) y[i] = 0.0f;
}

static void softmax_host(float* row, int out) {
    float maxv = row[0];
    for (int i = 1; i < out; ++i) if (row[i] > maxv) maxv = row[i];
    float sum = 0.0f;
    for (int i = 0; i < out; ++i) { row[i] = expf(row[i] - maxv); sum += row[i]; }
    for (int i = 0; i < out; ++i) row[i] /= sum;
}

static int gpu_forward_sample(struct FAHRENModel* cm, cublasHandle_t handle,
                              const float* x, float*** layer_outputs, size_t* layer_out_sizes) {
    size_t L = cm->layer_count;
    const float alpha = 1.0f, beta = 0.0f;

    *layer_outputs = (float**)calloc(L + 1, sizeof(float*));
    if (!*layer_outputs) return -1;

    (*layer_outputs)[0] = (float*)malloc(cm->input_dim * sizeof(float));
    if (!(*layer_outputs)[0]) return -1;
    memcpy((*layer_outputs)[0], x, cm->input_dim * sizeof(float));
    layer_out_sizes[0] = cm->input_dim;

    float* d_in = NULL;
    float* d_out = NULL;
    float* d_w = NULL;
    float* d_b = NULL;
    cudaMalloc(&d_in, cm->input_dim * sizeof(float));
    cudaMemcpy(d_in, x, cm->input_dim * sizeof(float), cudaMemcpyHostToDevice);

  for (size_t li = 0; li < L; ++li) {
        FAHRENLayer* layer = &cm->layers[li];
        FAHRENLayerParams* P = &cm->cache->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;

        cudaMalloc(&d_out, out * sizeof(float));
        cudaMalloc(&d_w, P->weight_count * sizeof(float));
        cudaMalloc(&d_b, out * sizeof(float));
        cudaMemcpy(d_w, P->weights, P->weight_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, P->biases, out * sizeof(float), cudaMemcpyHostToDevice);

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)out, 1, (int)in,
                    &alpha, d_w, (int)in, d_in, (int)in, &beta, d_out, (int)out);
        add_bias_kernel<<<(int)((out + 255) / 256), 256>>>(d_out, d_b, (int)out);

        (*layer_outputs)[li + 1] = (float*)malloc(out * sizeof(float));
        if (!(*layer_outputs)[li + 1]) return -1;

        if (layer->activation == FAHREN_LAYER_ACTIVATION_RELU) {
            relu_kernel<<<(int)((out + 255) / 256), 256>>>(d_out, (int)out);
        }

        cudaMemcpy((*layer_outputs)[li + 1], d_out, out * sizeof(float), cudaMemcpyDeviceToHost);
        if (layer->activation == FAHREN_LAYER_ACTIVATION_SOFTMAX) {
            softmax_host((*layer_outputs)[li + 1], (int)out);
        }
        layer_out_sizes[li + 1] = out;

        cudaFree(d_w);
        cudaFree(d_b);
        cudaFree(d_out);
        cudaFree(d_in);
        d_in = NULL;
        cudaMalloc(&d_in, out * sizeof(float));
        cudaMemcpy(d_in, (*layer_outputs)[li + 1], out * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (d_in) cudaFree(d_in);
    return 0;
}

int fahren_train_cuda(struct FAHRENModel* cm, const float* inputs, size_t sample_count,
                                 size_t input_dim, const int* labels, size_t num_classes,
                                 const char* weights_path, size_t epochs,
                                 const FAHRENTrainConfig* config) {
    (void)num_classes;
    if (!fahren_cuda_init()) return FAHREN_ERROR_UNSUPPORTED;

    int rc = fahren_weights_load(cm, weights_path);
    if (rc != FAHREN_SUCCESS) return rc;

    if (config->optimizer) {
        fahren_ensure_optimizer_states(cm, config->optimizer);
    }

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    size_t batch_size = config->batch_size ? config->batch_size : 32;
    if (batch_size > sample_count) batch_size = sample_count;

    size_t iteration = 0;

    for (size_t e = 0; e < epochs; ++e) {
        for (size_t batch_start = 0; batch_start < sample_count; batch_start += batch_size) {
            size_t batch_end = batch_start + batch_size;
            if (batch_end > sample_count) batch_end = sample_count;
            size_t this_batch = batch_end - batch_start;

            fahren_zero_layer_grads(cm);

            for (size_t s = batch_start; s < batch_end; ++s) {
                float** layer_outputs = NULL;
                size_t* layer_sizes = (size_t*)calloc(cm->layer_count + 1, sizeof(size_t));
                if (!layer_sizes) continue;

                if (gpu_forward_sample(cm, handle, &inputs[s * input_dim], &layer_outputs, layer_sizes) != 0) {
                    free(layer_sizes);
                    continue;
                }

                if (fahren_backward_accumulate(cm, layer_outputs, layer_sizes, labels[s]) != 0) {
                    fahren_free_layer_outputs(layer_outputs, cm->layer_count);
                    free(layer_sizes);
                    cublasDestroy(handle);
                    return FAHREN_ERROR_PROCESSING_FAILED;
                }

                fahren_free_layer_outputs(layer_outputs, cm->layer_count);
                free(layer_sizes);
            }

            fahren_apply_layer_gradients(cm, config, this_batch, iteration++);
        }

        rc = fahren_weights_flush(cm, weights_path);
        if (rc != FAHREN_SUCCESS) {
            cublasDestroy(handle);
            return rc;
        }
    }

    cublasDestroy(handle);
    return FAHREN_SUCCESS;
}
