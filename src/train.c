#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include <nova/errors.h>
#include "internal.h"

static inline float act_fwd(int act, float x) {
    switch (act) {
        case NOVA_ACTIVATION_RELU:    return x > 0.0f ? x : 0.0f;
        case NOVA_ACTIVATION_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case NOVA_ACTIVATION_TANH:    return tanhf(x);
        case NOVA_ACTIVATION_LINEAR:
        default:                      return x;
    }
}

static inline float act_bwd(int act, float y) {
    switch (act) {
        case NOVA_ACTIVATION_RELU:    return (y > 0.0f) ? 1.0f : 0.0f;
        case NOVA_ACTIVATION_SIGMOID: return y * (1.0f - y);
        case NOVA_ACTIVATION_TANH:    return 1.0f - y * y;
        case NOVA_ACTIVATION_LINEAR:
        default:                      return 1.0f;
    }
}

static void apply_softmax(float* logits, size_t n) {
    float maxv = logits[0];
    for (size_t i = 1; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { logits[i] = expf(logits[i] - maxv); sum += logits[i]; }
    for (size_t i = 0; i < n; ++i) logits[i] /= sum;
}

void nova_zero_grads(NOVAModel* model) {
    NOVAWeightCache* cache = model->cache;
    size_t L = model->layer_count;
    for (size_t i = 0; i < L; ++i) {
        memset(cache->layers[i].grad_weights, 0, cache->layers[i].weight_count * sizeof(float));
        memset(cache->layers[i].grad_biases, 0, cache->layers[i].bias_count * sizeof(float));
    }
}

int nova_forward(NOVAModel* model, const float* x,
                 float*** layer_outputs, size_t* layer_out_sizes) {
    size_t L = model->layer_count;
    NOVAWeightCache* cache = model->cache;

    *layer_outputs = (float**)calloc(L + 1, sizeof(float*));
    if (!*layer_outputs) return -1;

    (*layer_outputs)[0] = (float*)malloc(model->input_dim * sizeof(float));
    if (!(*layer_outputs)[0]) return -1;
    memcpy((*layer_outputs)[0], x, model->input_dim * sizeof(float));
    layer_out_sizes[0] = model->input_dim;

    float* preact = (float*)malloc(model->layers[0].output_size * sizeof(float));
    if (!preact) return -1;

    for (size_t i = 0; i < L; ++i) {
        NOVALayer* layer = &model->layers[i];
        NOVALayerParams* P = &cache->layers[i];
        size_t in = layer->input_size;
        size_t out = layer->output_size;
        float* prev = (*layer_outputs)[i];

        nova_gemm(0, 0, out, 1, in, 1.0f, P->weights, in, prev, 1, 0.0f, preact, 1);
        for (size_t o = 0; o < out; ++o) preact[o] += P->biases[o];

        float* cur = (float*)malloc(out * sizeof(float));
        if (!cur) return -1;

        if (layer->activation == NOVA_ACTIVATION_SOFTMAX) {
            memcpy(cur, preact, out * sizeof(float));
        } else {
            for (size_t o = 0; o < out; ++o)
                cur[o] = act_fwd(layer->activation, preact[o]);
        }

        (*layer_outputs)[i + 1] = cur;
        layer_out_sizes[i + 1] = out;
    }

    free(preact);

    NOVALayer* last = &model->layers[L - 1];
    if (last->activation == NOVA_ACTIVATION_SOFTMAX)
        apply_softmax((*layer_outputs)[L], layer_out_sizes[L]);

    return 0;
}

void nova_free_outputs(float** outs, size_t L) {
    if (!outs) return;
    for (size_t i = 0; i <= L; ++i) free(outs[i]);
    free(outs);
}

int nova_backward(NOVAModel* model, float** layer_outputs, size_t* layer_out_sizes,
                  int label) {
    size_t L = model->layer_count;
    NOVAWeightCache* cache = model->cache;

    float** deltas = (float**)calloc(L, sizeof(float*));
    if (!deltas) return -1;

    size_t outn = layer_out_sizes[L];
    deltas[L - 1] = (float*)malloc(outn * sizeof(float));
    if (!deltas[L - 1]) { free(deltas); return -1; }

    for (size_t i = 0; i < outn; ++i) deltas[L - 1][i] = layer_outputs[L][i];
    if (label >= 0 && (size_t)label < outn) deltas[L - 1][label] -= 1.0f;

    for (size_t li = L - 1; li > 0; --li) {
        NOVALayer* layer = &model->layers[li];
        NOVALayerParams* P = &cache->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;
        float* delta_next = deltas[li];

        for (size_t o = 0; o < out; ++o) {
            cache->layers[li].grad_biases[o] += delta_next[o];
            const float* prev = layer_outputs[li];
            float* wrow = &P->grad_weights[o * in];
            for (size_t ii = 0; ii < in; ++ii)
                wrow[ii] += delta_next[o] * prev[ii];
        }

        float* delta_cur = (float*)calloc(in, sizeof(float));
        if (!delta_cur) {
            for (size_t k = 0; k < L; ++k) free(deltas[k]);
            free(deltas); return -1;
        }

        for (size_t ii = 0; ii < in; ++ii) {
            float sum = 0.0f;
            for (size_t o = 0; o < out; ++o)
                sum += P->weights[o * in + ii] * delta_next[o];
            float y = layer_outputs[li][ii];
            sum *= act_bwd(model->layers[li - 1].activation, y);
            delta_cur[ii] = sum;
        }
        deltas[li - 1] = delta_cur;
    }

    {
        size_t li = 0;
        NOVALayer* layer = &model->layers[li];
        NOVALayerParams* P = &cache->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;
        float* delta_next = deltas[li];

        for (size_t o = 0; o < out; ++o) {
            P->grad_biases[o] += delta_next[o];
            const float* prev = layer_outputs[li];
            float* wrow = &P->grad_weights[o * in];
            for (size_t ii = 0; ii < in; ++ii)
                wrow[ii] += delta_next[o] * prev[ii];
        }
    }

    for (size_t k = 0; k < L; ++k) free(deltas[k]);
    free(deltas);
    return 0;
}

void nova_ensure_opt_states(NOVAModel* model, const NOVAOptimizer* opt) {
    if (!opt) return;
    (void)model; /* placeholder for optimizer state initialization */
}

void nova_apply_grads(NOVAModel* model, const NOVATrainConfig* config,
                      size_t batch_size, size_t iteration) {
    (void)iteration;
    float inv_batch = 1.0f / (float)batch_size;
    NOVAOptimizer* opt = config->optimizer;

    for (size_t i = 0; i < model->layer_count; ++i) {
        NOVALayerParams* P = &model->cache->layers[i];
        size_t wc = P->weight_count;
        size_t bc = P->bias_count;
        size_t k;

        for (k = 0; k < wc; ++k) P->grad_weights[k] *= inv_batch;
        for (k = 0; k < bc; ++k) P->grad_biases[k] *= inv_batch;

        if (opt) {
            /* Simple SGD fallback for now */
            float lr = opt->learning_rate;
            for (k = 0; k < wc; ++k) P->weights[k] -= lr * P->grad_weights[k];
            for (k = 0; k < bc; ++k) P->biases[k] -= lr * P->grad_biases[k];
        } else {
            float lr = config->learning_rate;
            for (k = 0; k < wc; ++k) P->weights[k] -= lr * P->grad_weights[k];
            for (k = 0; k < bc; ++k) P->biases[k] -= lr * P->grad_biases[k];
        }
    }
    model->cache->dirty = 1;
}

NOVA_Status nova_train_cpu(NOVAModel* model, const float* inputs, size_t sample_count,
                           size_t input_dim, const int* labels, size_t num_classes,
                           const char* path, size_t epochs,
                           const NOVATrainConfig* config) {
    (void)num_classes;
    if (!model || !inputs || !labels || !path || !config)
        return NOVA_ERROR_INVALID_ARGUMENT;
    if (!model->finalized || model->input_dim != input_dim)
        return NOVA_ERROR_INVALID_ARGUMENT;

    NOVA_Status rc = nova_weights_load(model, path);
    if (rc != NOVA_SUCCESS) return rc;

    if (config->optimizer)
        nova_ensure_opt_states(model, config->optimizer);

    size_t batch_size = config->batch_size ? config->batch_size : 32;
    if (batch_size > sample_count) batch_size = sample_count;

    size_t iteration = 0;

    for (size_t e = 0; e < epochs; ++e) {
        double epoch_loss = 0.0;

        for (size_t batch_start = 0; batch_start < sample_count; batch_start += batch_size) {
            size_t batch_end = batch_start + batch_size;
            if (batch_end > sample_count) batch_end = sample_count;
            size_t this_batch = batch_end - batch_start;

            nova_zero_grads(model);

            for (size_t s = batch_start; s < batch_end; ++s) {
                const float* x = &inputs[s * input_dim];
                int y = labels[s];

                float** layer_outputs = NULL;
                size_t* layer_sizes = (size_t*)calloc(model->layer_count + 1, sizeof(size_t));
                if (!layer_sizes) return NOVA_ERROR_OUT_OF_MEMORY;

                if (nova_forward(model, x, &layer_outputs, layer_sizes) != 0) {
                    free(layer_sizes);
                    return NOVA_ERROR_PROCESSING_FAILED;
                }

                float py = 1e-9f;
                if (y >= 0 && (size_t)y < layer_sizes[model->layer_count])
                    py = layer_outputs[model->layer_count][y];
                epoch_loss += -logf(py);

                if (nova_backward(model, layer_outputs, layer_sizes, y) != 0) {
                    nova_free_outputs(layer_outputs, model->layer_count);
                    free(layer_sizes);
                    return NOVA_ERROR_PROCESSING_FAILED;
                }

                nova_free_outputs(layer_outputs, model->layer_count);
                free(layer_sizes);
            }

            nova_apply_grads(model, config, this_batch, iteration++);
        }

        rc = nova_weights_flush(model, path);
        if (rc != NOVA_SUCCESS) return rc;
    }

    return NOVA_SUCCESS;
}

static int argmax(const float* out, size_t n, int* predicted_class) {
    if (!out || !predicted_class || n == 0) return -1;
    int best = 0;
    float bestv = out[0];
    for (size_t i = 1; i < n; ++i) {
        if (out[i] > bestv) { bestv = out[i]; best = (int)i; }
    }
    *predicted_class = best;
    return 0;
}

NOVA_Status nova_predict(NOVAModel* model, const char* path,
                         const float* input, size_t input_dim, int* class_out) {
    if (!model || !path || !input || !class_out)
        return NOVA_ERROR_INVALID_ARGUMENT;
    if (!model->finalized || model->input_dim != input_dim)
        return NOVA_ERROR_INVALID_ARGUMENT;

    NOVA_Status rc = nova_weights_load(model, path);
    if (rc != NOVA_SUCCESS) return rc;

    float** layer_outputs = NULL;
    size_t* layer_sizes = (size_t*)calloc(model->layer_count + 1, sizeof(size_t));
    if (!layer_sizes) return NOVA_ERROR_OUT_OF_MEMORY;

    if (nova_forward(model, input, &layer_outputs, layer_sizes) != 0) {
        free(layer_sizes);
        return NOVA_ERROR_PROCESSING_FAILED;
    }

    argmax(layer_outputs[model->layer_count], layer_sizes[model->layer_count], class_out);

    nova_free_outputs(layer_outputs, model->layer_count);
    free(layer_sizes);
    return NOVA_SUCCESS;
}

NOVA_Status nova_evaluate(NOVAModel* model, const char* path,
                          const float* inputs, const int* labels,
                          size_t sample_count, size_t input_dim, float* accuracy) {
    if (!model || !path || !inputs || !labels || !accuracy)
        return NOVA_ERROR_INVALID_ARGUMENT;
    if (!model->finalized || model->input_dim != input_dim || sample_count == 0)
        return NOVA_ERROR_INVALID_ARGUMENT;

    NOVA_Status rc = nova_weights_load(model, path);
    if (rc != NOVA_SUCCESS) return rc;

    size_t correct = 0;
    for (size_t s = 0; s < sample_count; ++s) {
        int pred = 0;
        rc = nova_predict(model, path, &inputs[s * input_dim], input_dim, &pred);
        if (rc != NOVA_SUCCESS) return rc;
        if (pred == labels[s]) ++correct;
    }

    *accuracy = (float)correct / (float)sample_count;
    return NOVA_SUCCESS;
}

NOVATrainConfig nova_train_config_default(float learning_rate) {
    NOVATrainConfig cfg;
    cfg.batch_size = 32;
    cfg.learning_rate = learning_rate;
    cfg.optimizer = NULL;
    cfg.device = NOVA_DEVICE_DEFAULT;
    return cfg;
}

static int resolve_device(const NOVATrainConfig* config) {
    if (config && config->device >= 0)
        return config->device;
    return nova_device_get();
}

NOVA_Status nova_train_with_config(NOVAModel* model, const float* inputs,
                                   size_t sample_count, size_t input_dim,
                                   const int* labels, size_t num_classes,
                                   const char* path, size_t epochs,
                                   const NOVATrainConfig* config) {
    if (!config) return NOVA_ERROR_INVALID_ARGUMENT;

    int dev = resolve_device(config);
#ifdef NOVA_ENABLE_CUDA
    if (dev == NOVA_DEVICE_CUDA)
        return nova_train_cuda(model, inputs, sample_count, input_dim, labels,
                               num_classes, path, epochs, config);
#endif
    return nova_train_cpu(model, inputs, sample_count, input_dim, labels,
                          num_classes, path, epochs, config);
}

NOVA_Status nova_train(NOVAModel* model, const float* inputs, size_t sample_count,
                       size_t input_dim, const int* labels, size_t num_classes,
                       const char* path, size_t epochs, float learning_rate) {
    NOVATrainConfig cfg = nova_train_config_default(learning_rate);
    return nova_train_with_config(model, inputs, sample_count, input_dim, labels,
                                  num_classes, path, epochs, &cfg);
}
