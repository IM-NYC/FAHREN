#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include <fahren/errors.h>
#include "internal.h"

static inline float act_forward(int act, float x) {
    switch (act) {
        case FAHREN_LAYER_ACTIVATION_RELU: return x > 0.0f ? x : 0.0f;
        case FAHREN_LAYER_ACTIVATION_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case FAHREN_LAYER_ACTIVATION_TANH: return tanhf(x);
        default: return x;
    }
}

static inline float act_backward(int act, float y) {
    switch (act) {
        case FAHREN_LAYER_ACTIVATION_RELU: return (y > 0.0f) ? 1.0f : 0.0f;
        case FAHREN_LAYER_ACTIVATION_SIGMOID: return y * (1.0f - y);
        case FAHREN_LAYER_ACTIVATION_TANH: return 1.0f - y * y;
        default: return 1.0f;
    }
}

static void apply_softmax(float* logits, size_t n) {
    float maxv = logits[0];
    for (size_t i = 1; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { logits[i] = expf(logits[i] - maxv); sum += logits[i]; }
    for (size_t i = 0; i < n; ++i) logits[i] /= sum;
}

void fahren_zero_layer_grads(struct FAHRENModel* cm) {
    FAHRENWeightCache* cache = cm->cache;
    size_t L = cm->layer_count;
    for (size_t i = 0; i < L; ++i) {
        memset(cache->layers[i].grad_weights, 0, cache->layers[i].weight_count * sizeof(float));
        memset(cache->layers[i].grad_biases, 0, cache->layers[i].bias_count * sizeof(float));
    }
}

int fahren_forward_cached(FAHRENModel* cm, const float* x,
                                 float*** layer_outputs, size_t* layer_out_sizes) {
    size_t L = cm->layer_count;
    FAHRENWeightCache* cache = cm->cache;

    *layer_outputs = (float**)calloc(L + 1, sizeof(float*));
    if (!*layer_outputs) return -1;

    (*layer_outputs)[0] = (float*)malloc(cm->input_dim * sizeof(float));
    if (!(*layer_outputs)[0]) return -1;
    memcpy((*layer_outputs)[0], x, cm->input_dim * sizeof(float));
    layer_out_sizes[0] = cm->input_dim;

    float* preact = (float*)malloc(cm->layers[0].output_size * sizeof(float));
    if (!preact) return -1;

    for (size_t i = 0; i < L; ++i) {
        FAHRENLayer* layer = &cm->layers[i];
        FAHRENLayerParams* P = &cache->layers[i];
        size_t in = layer->input_size;
        size_t out = layer->output_size;
        float* prev = (*layer_outputs)[i];

        fahren_gemm(0, 0, out, 1, in, 1.0f, P->weights, in, prev, 1, 0.0f, preact, 1);
        for (size_t o = 0; o < out; ++o) {
            preact[o] += P->biases[o];
        }

        float* cur = (float*)malloc(out * sizeof(float));
        if (!cur) return -1;

        if (layer->activation == FAHREN_LAYER_ACTIVATION_SOFTMAX) {
            memcpy(cur, preact, out * sizeof(float));
        } else {
            for (size_t o = 0; o < out; ++o) {
                cur[o] = act_forward(layer->activation, preact[o]);
            }
        }

        (*layer_outputs)[i + 1] = cur;
        layer_out_sizes[i + 1] = out;
    }

    free(preact);

    FAHRENLayer* last = &cm->layers[L - 1];
    if (last->activation == FAHREN_LAYER_ACTIVATION_SOFTMAX) {
        apply_softmax((*layer_outputs)[L], layer_out_sizes[L]);
    }

    return 0;
}

void fahren_free_layer_outputs(float** outs, size_t L) {
    if (!outs) return;
    for (size_t i = 0; i <= L; ++i) free(outs[i]);
    free(outs);
}

int fahren_backward_accumulate(FAHRENModel* cm, float** layer_outputs, size_t* layer_out_sizes,
                               int label) {
    size_t L = cm->layer_count;
    FAHRENWeightCache* cache = cm->cache;

    float** deltas = (float**)calloc(L, sizeof(float*));
    if (!deltas) return -1;

    size_t outn = layer_out_sizes[L];
    deltas[L - 1] = (float*)malloc(outn * sizeof(float));
    if (!deltas[L - 1]) { free(deltas); return -1; }

    for (size_t i = 0; i < outn; ++i) deltas[L - 1][i] = layer_outputs[L][i];
    if (label >= 0 && (size_t)label < outn) deltas[L - 1][label] -= 1.0f;

    for (size_t li = L - 1; li > 0; --li) {
        FAHRENLayer* layer = &cm->layers[li];
        FAHRENLayerParams* P = &cache->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;
        float* delta_next = deltas[li];

        for (size_t o = 0; o < out; ++o) {
            cache->layers[li].grad_biases[o] += delta_next[o];
            const float* prev = layer_outputs[li];
            float* wrow = &P->grad_weights[o * in];
            for (size_t ii = 0; ii < in; ++ii) {
                wrow[ii] += delta_next[o] * prev[ii];
            }
        }

        float* delta_cur = (float*)calloc(in, sizeof(float));
        if (!delta_cur) {
            for (size_t k = 0; k < L; ++k) free(deltas[k]);
            free(deltas);
            return -1;
        }

        for (size_t ii = 0; ii < in; ++ii) {
            float sum = 0.0f;
            for (size_t o = 0; o < out; ++o) {
                sum += P->weights[o * in + ii] * delta_next[o];
            }
            float y = layer_outputs[li][ii];
            sum *= act_backward(cm->layers[li - 1].activation, y);
            delta_cur[ii] = sum;
        }

        deltas[li - 1] = delta_cur;
    }

    {
        size_t li = 0;
        FAHRENLayer* layer = &cm->layers[li];
        FAHRENLayerParams* P = &cache->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;
        float* delta_next = deltas[li];

        for (size_t o = 0; o < out; ++o) {
            P->grad_biases[o] += delta_next[o];
            const float* prev = layer_outputs[li];
            float* wrow = &P->grad_weights[o * in];
            for (size_t ii = 0; ii < in; ++ii) {
                wrow[ii] += delta_next[o] * prev[ii];
            }
        }
    }

    for (size_t k = 0; k < L; ++k) free(deltas[k]);
    free(deltas);
    return 0;
}

void fahren_ensure_optimizer_states(struct FAHRENModel* cm, const FAHRENOptimizer* opt) {
    if (!opt) return;
    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayerParams* P = &cm->cache->layers[i];
        if (!P->opt_state_w) {
            P->opt_state_w = fahren_optimizer_state_create(opt, P->weight_count);
        }
        if (!P->opt_state_b) {
            P->opt_state_b = fahren_optimizer_state_create(opt, P->bias_count);
        }
    }
}

void fahren_apply_layer_gradients(struct FAHRENModel* cm, const FAHRENTrainConfig* config,
                            size_t batch_size, size_t iteration) {
    float inv_batch = 1.0f / (float)batch_size;
    FAHRENOptimizer* opt = config->optimizer;

    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayerParams* P = &cm->cache->layers[i];
        size_t wc = P->weight_count;
        size_t bc = P->bias_count;
        size_t k;

        for (k = 0; k < wc; ++k) P->grad_weights[k] *= inv_batch;
        for (k = 0; k < bc; ++k) P->grad_biases[k] *= inv_batch;

        if (opt) {
            FAHRENOptimizer opt_mut = *opt;
            fahren_optimizer_update(&opt_mut, P->opt_state_w, P->weights, P->grad_weights, wc, iteration);
            fahren_optimizer_update(&opt_mut, P->opt_state_b, P->biases, P->grad_biases, bc, iteration);
        } else {
            float lr = config->learning_rate;
            for (k = 0; k < wc; ++k) P->weights[k] -= lr * P->grad_weights[k];
            for (k = 0; k < bc; ++k) P->biases[k] -= lr * P->grad_biases[k];
        }
    }
    cm->cache->dirty = 1;
}

int fahren_train_cpu(struct FAHRENModel* cm, const float* inputs, size_t sample_count,
                     size_t input_dim, const int* labels, size_t num_classes,
                     const char* weights_path, size_t epochs,
                     const FAHRENTrainConfig* config) {
    (void)num_classes;
    if (!cm || !inputs || !labels || !weights_path || !config) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
    if (!cm->finalized || cm->input_dim != input_dim) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }

    int rc = fahren_weights_load(cm, weights_path);
    if (rc != FAHREN_SUCCESS) return rc;

    if (config->optimizer) {
        fahren_ensure_optimizer_states(cm, config->optimizer);
    }

    size_t batch_size = config->batch_size ? config->batch_size : 32;
    if (batch_size > sample_count) batch_size = sample_count;

    size_t iteration = 0;

    for (size_t e = 0; e < epochs; ++e) {
        double epoch_loss = 0.0;

        for (size_t batch_start = 0; batch_start < sample_count; batch_start += batch_size) {
            size_t batch_end = batch_start + batch_size;
            if (batch_end > sample_count) batch_end = sample_count;
            size_t this_batch = batch_end - batch_start;

            fahren_zero_layer_grads(cm);

            for (size_t s = batch_start; s < batch_end; ++s) {
                const float* x = &inputs[s * input_dim];
                int y = labels[s];

                float** layer_outputs = NULL;
                size_t* layer_sizes = (size_t*)calloc(cm->layer_count + 1, sizeof(size_t));
                if (!layer_sizes) return FAHREN_ERROR_OUT_OF_MEMORY;

                if (fahren_forward_cached(cm, x, &layer_outputs, layer_sizes) != 0) {
                    free(layer_sizes);
                    return FAHREN_ERROR_PROCESSING_FAILED;
                }

                float py = 1e-9f;
                if (y >= 0 && (size_t)y < layer_sizes[cm->layer_count]) {
                    py = layer_outputs[cm->layer_count][y];
                }
                epoch_loss += -logf(py);

                if (fahren_backward_accumulate(cm, layer_outputs, layer_sizes, y) != 0) {
                    fahren_free_layer_outputs(layer_outputs, cm->layer_count);
                    free(layer_sizes);
                    return FAHREN_ERROR_PROCESSING_FAILED;
                }

                fahren_free_layer_outputs(layer_outputs, cm->layer_count);
                free(layer_sizes);
            }

            fahren_apply_layer_gradients(cm, config, this_batch, iteration++);
        }

        rc = fahren_weights_flush(cm, weights_path);
        if (rc != FAHREN_SUCCESS) return rc;

#if FAHREN_VERBOSE
        fprintf(stdout, "Epoch %zu/%zu - loss: %.6f\n", e + 1, epochs,
                epoch_loss / (double)sample_count);
#endif
    }

    return FAHREN_SUCCESS;
}

FAHRENTrainConfig fahren_train_config_default(float learning_rate) {
    FAHRENTrainConfig cfg;
    cfg.batch_size = 32;
    cfg.learning_rate = learning_rate;
    cfg.optimizer = NULL;
    cfg.device = FAHREN_DEVICE_DEFAULT;
    return cfg;
}

FAHRENTrainConfig fahren_train_config_cuda(float learning_rate) {
    FAHRENTrainConfig cfg = fahren_train_config_default(learning_rate);
    cfg.device = FAHREN_DEVICE_CUDA;
    cfg.batch_size = 64;
    return cfg;
}

static int train_dispatch(FAHRENModel* cm, const float* inputs, size_t sample_count,
                          size_t input_dim, const int* labels, size_t num_classes,
                          const char* weights_path, size_t epochs,
                          const FAHRENTrainConfig* config) {
    int device = fahren_train_resolve_device(config);

    if (device == FAHREN_DEVICE_CUDA) {
#ifdef FAHREN_ENABLE_CUDA
        if (!fahren_cuda_available()) {
            return FAHREN_ERROR_UNSUPPORTED;
        }
        return fahren_train_cuda(cm, inputs, sample_count, input_dim, labels, num_classes,
                                 weights_path, epochs, config);
#else
        return FAHREN_ERROR_UNSUPPORTED;
#endif
    }

    return fahren_train_cpu(cm, inputs, sample_count, input_dim, labels, num_classes,
                            weights_path, epochs, config);
}

int fahren_train_with_config(FAHRENModel* cm, const float* inputs, size_t sample_count,
                             size_t input_dim, const int* labels, size_t num_classes,
                             const char* weights_path, size_t epochs,
                             const FAHRENTrainConfig* config) {
    if (!config) return FAHREN_ERROR_INVALID_ARGUMENT;
    if (config->device == FAHREN_DEVICE_CUDA && !fahren_cuda_available()) {
        return FAHREN_ERROR_UNSUPPORTED;
    }
    return train_dispatch(cm, inputs, sample_count, input_dim, labels, num_classes,
                          weights_path, epochs, config);
}

int fahren_train(FAHRENModel* cm, const float* inputs, size_t sample_count, size_t input_dim,
                 const int* labels, size_t num_classes, const char* weights_path,
                 size_t epochs, float learning_rate) {
    FAHRENTrainConfig cfg = fahren_train_config_default(learning_rate);
    return fahren_train_with_config(cm, inputs, sample_count, input_dim, labels, num_classes,
                                    weights_path, epochs, &cfg);
}

static int argmax_output(const float* out, size_t n, int* predicted_class) {
    if (!out || !predicted_class || n == 0) return FAHREN_ERROR_INVALID_ARGUMENT;
    int best = 0;
    float bestv = out[0];
    for (size_t i = 1; i < n; ++i) {
        if (out[i] > bestv) { bestv = out[i]; best = (int)i; }
    }
    *predicted_class = best;
    return FAHREN_SUCCESS;
}

int fahren_predict(FAHRENModel* cm, const char* weights_path, const float* input,
                   size_t input_dim, int* predicted_class) {
    if (!cm || !weights_path || !input || !predicted_class) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
    if (!cm->finalized || cm->input_dim != input_dim) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }

    int rc = fahren_weights_load(cm, weights_path);
    if (rc != FAHREN_SUCCESS) return rc;

    float** layer_outputs = NULL;
    size_t* layer_sizes = (size_t*)calloc(cm->layer_count + 1, sizeof(size_t));
    if (!layer_sizes) return FAHREN_ERROR_OUT_OF_MEMORY;

    if (fahren_forward_cached(cm, input, &layer_outputs, layer_sizes) != 0) {
        free(layer_sizes);
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    rc = argmax_output(layer_outputs[cm->layer_count], layer_sizes[cm->layer_count],
                       predicted_class);

    fahren_free_layer_outputs(layer_outputs, cm->layer_count);
    free(layer_sizes);
    return rc;
}

int fahren_evaluate(FAHRENModel* cm, const char* weights_path,
                    const float* inputs, const int* labels,
                    size_t sample_count, size_t input_dim, float* accuracy_out) {
    if (!cm || !weights_path || !inputs || !labels || !accuracy_out) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
    if (!cm->finalized || cm->input_dim != input_dim || sample_count == 0) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }

    int rc = fahren_weights_load(cm, weights_path);
    if (rc != FAHREN_SUCCESS) return rc;

    size_t correct = 0;
    for (size_t s = 0; s < sample_count; ++s) {
        int pred = 0;
        rc = fahren_predict(cm, weights_path, &inputs[s * input_dim], input_dim, &pred);
        if (rc != FAHREN_SUCCESS) return rc;
        if (pred == labels[s]) ++correct;
    }

    *accuracy_out = (float)correct / (float)sample_count;
    return FAHREN_SUCCESS;
}
