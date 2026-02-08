/*
 * FAHREN Training Engine
 * 
 * This module implements the core training algorithm for neural networks:
 * - Forward pass: compute predictions through sequential layers
 * - Loss computation: cross-entropy for classification tasks
 * - Backward pass: backpropagation to compute gradients
 * - Parameter updates: gradient descent optimization
 * - Activation functions: ReLU, Sigmoid, Tanh, Softmax
 * 
 * TRAINING FLOW:
 * 1. Load model weights from file
 * 2. For each epoch:
 *    a. For each training sample:
 *       - Forward pass: compute layer-by-layer activations
 *       - Compute loss: cross-entropy of predicted vs true class
 *       - Backward pass: compute gradients w.r.t. all parameters
 *       - Update: apply gradient descent to weights and biases
 *    b. Save updated weights to file
 * 3. Return final loss/accuracy metrics
 * 
 * OPTIMIZATION:
 * - Currently uses vanilla SGD (stochastic gradient descent)
 * - Learning rate: constant throughout training (no scheduling)
 * - Future: Adam, RMSprop, momentum variants
 * 
 * NUMERICAL CONSIDERATIONS:
 * - Xavier/Glorot initialization for stable training
 * - Small learning rate (0.01) to avoid instability
 * - Cross-entropy loss for probabilistic outputs
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "internal.h"

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================
 * 
 * Forward pass kernels for each activation type
 */
static inline float act_forward(int act, float x) {
    switch (act) {
        case FAHREN_LAYER_ACTIVATION_RELU: return x > 0.0f ? x : 0.0f;
        case FAHREN_LAYER_ACTIVATION_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case FAHREN_LAYER_ACTIVATION_TANH: return tanhf(x);
        case FAHREN_LAYER_ACTIVATION_SOFTMAX: /* handled separately */ return x;
        default: return x;
    }
}

/* Activation function derivatives w.r.t. output
 * Used in backward pass chain rule: d_loss/d_input = d_loss/d_output * d_output/d_input
 */
static inline float act_backward(int act, float y /* output of act */) {
    switch (act) {
        case FAHREN_LAYER_ACTIVATION_RELU: return (y > 0.0f) ? 1.0f : 0.0f;
        case FAHREN_LAYER_ACTIVATION_SIGMOID: return y * (1.0f - y);
        case FAHREN_LAYER_ACTIVATION_TANH: return 1.0f - y * y;
        case FAHREN_LAYER_ACTIVATION_SOFTMAX: return 1.0f; /* use cross-entropy simplification */
        default: return 1.0f;
    }
}


/* Read/write helpers */
static int read_weights(FILE* f, long offset, float* dst, size_t n) {
    if (fseek(f, offset, SEEK_SET) != 0) return -1;
    return (fread(dst, sizeof(float), n, f) == n) ? 0 : -1;
}
static int write_weights(FILE* f, long offset, const float* src, size_t n) {
    if (fseek(f, offset, SEEK_SET) != 0) return -1;
    return (fwrite(src, sizeof(float), n, f) == n) ? 0 : -1;
}

/* Forward pass for a single sample */
static int forward_sample(FAHRENModel* cm, FILE* f, const float* x,
                          float*** layer_outputs, size_t* layer_out_sizes) {
    size_t L = cm->layer_count;
    *layer_outputs = (float**)calloc(L + 1, sizeof(float*));
    if (!*layer_outputs) return -1;

    (*layer_outputs)[0] = (float*)malloc(cm->input_dim * sizeof(float));
    if (!(*layer_outputs)[0]) return -1;
    memcpy((*layer_outputs)[0], x, cm->input_dim * sizeof(float));
    layer_out_sizes[0] = cm->input_dim;

    for (size_t i = 0; i < L; ++i) {
        FAHRENLayer* layer = &cm->layers[i];
        size_t in = layer->input_size;
        size_t out = layer->output_size;

        float* prev = (*layer_outputs)[i];
        float* cur = (float*)calloc(out, sizeof(float));
        if (!cur) return -1;

        /* Read weights and biases */
        size_t wcount = in * out;
        float* W = (float*)malloc(wcount * sizeof(float));
        float* b = (float*)malloc(out * sizeof(float));
        if (!W || !b) return -1;
        if (read_weights(f, layer->weights_offset, W, wcount) != 0) return -1;
        if (read_weights(f, layer->bias_offset, b, out) != 0) return -1;

        /* y = act(W * x + b) */
        for (size_t o = 0; o < out; ++o) {
            float sum = b[o];
            const float* wrow = &W[o * in];
            for (size_t ii = 0; ii < in; ++ii) sum += wrow[ii] * prev[ii];
            cur[o] = act_forward(layer->activation, sum);
        }

        free(W); free(b);
        (*layer_outputs)[i+1] = cur;
        layer_out_sizes[i+1] = out;
    }

    /* Softmax on last layer if requested */
    FAHRENLayer* last = &cm->layers[L-1];
    if (last->activation == FAHREN_LAYER_ACTIVATION_SOFTMAX) {
        float* logits = (*layer_outputs)[L];
        size_t n = layer_out_sizes[L];
        float maxv = logits[0];
        for (size_t i = 1; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) { logits[i] = expf(logits[i] - maxv); sum += logits[i]; }
        for (size_t i = 0; i < n; ++i) logits[i] /= sum;
    }

    return 0;
}

static void free_layer_outputs(float** outs, size_t L) {
    if (!outs) return;
    for (size_t i = 0; i <= L; ++i) free(outs[i]);
    free(outs);
}

/* Backward + update for a single (x, label). Cross-entropy loss. */
static int backward_update(FAHRENModel* cm, FILE* f, float** layer_outputs,
                           size_t* layer_out_sizes, int label, float lr) {
    size_t L = cm->layer_count;

    /* Allocate deltas per layer output */
    float** deltas = (float**)calloc(L, sizeof(float*));
    if (!deltas) return -1;

    /* Output delta: softmax + CE => y - onehot(label) */
    size_t outn = layer_out_sizes[L];
    deltas[L-1] = (float*)calloc(outn, sizeof(float));
    if (!deltas[L-1]) return -1;

    for (size_t i = 0; i < outn; ++i) deltas[L-1][i] = layer_outputs[L][i];
    if (label >= 0 && (size_t)label < outn) deltas[L-1][label] -= 1.0f;

    /* Backpropagate to earlier layers */
    for (size_t li = L - 1; li > 0; --li) {
        FAHRENLayer* layer = &cm->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;

        float* delta_next = deltas[li];
        float* delta_cur = (float*)calloc(in, sizeof(float));
        if (!delta_cur) return -1;

        /* Read weights */
        size_t wcount = in * out;
        float* W = (float*)malloc(wcount * sizeof(float));
        if (!W) return -1;
        if (read_weights(f, layer->weights_offset, W, wcount) != 0) { free(W); return -1; }

        /* delta_cur = (W^T * delta_next) * act'(preact) ; we approximate act'(y) using output y */
        for (size_t i = 0; i < in; ++i) {
            float sum = 0.0f;
            for (size_t o = 0; o < out; ++o) sum += W[o * in + i] * delta_next[o];
            float y = layer_outputs[li][i];
            sum *= act_backward(cm->layers[li-1].activation, y); /* previous layer activation derivative */
            delta_cur[i] = sum;
        }
        free(W);
        deltas[li-1] = delta_cur;
    }

    /* Apply weight and bias updates */
    for (size_t li = 0; li < L; ++li) {
        FAHRENLayer* layer = &cm->layers[li];
        size_t in = layer->input_size;
        size_t out = layer->output_size;

        float* prev = layer_outputs[li];
        float* dY = deltas[li];

        size_t wcount = in * out;
        float* W = (float*)malloc(wcount * sizeof(float));
        float* b = (float*)malloc(out * sizeof(float));
        if (!W || !b) { free(W); free(b); return -1; }
        if (read_weights(f, layer->weights_offset, W, wcount) != 0) { free(W); free(b); return -1; }
        if (read_weights(f, layer->bias_offset, b, out) != 0) { free(W); free(b); return -1; }

        /* Gradient step */
        for (size_t o = 0; o < out; ++o) {
            float gbo = dY[o];
            b[o] -= lr * gbo;
            float* wrow = &W[o * in];
            for (size_t ii = 0; ii < in; ++ii) {
                wrow[ii] -= lr * gbo * prev[ii];
            }
        }

        if (write_weights(f, layer->weights_offset, W, wcount) != 0) { free(W); free(b); return -1; }
        if (write_weights(f, layer->bias_offset, b, out) != 0) { free(W); free(b); return -1; }
        free(W); free(b);
    }

    for (size_t i = 0; i < L; ++i) free(deltas[i]);
    free(deltas);
    return 0;
}

int fahren_train(FAHRENModel* cm,
                 const float* inputs, size_t sample_count, size_t input_dim,
                 const int* labels, size_t num_classes,
                 const char* weights_path, size_t epochs, float learning_rate) {
    if (!cm || !inputs || !labels || !weights_path) return FAHREN_ERROR_INVALID_ARGUMENT;
    if (!cm->finalized) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Model must be finalized to a file before training.\n");
        #endif
        return FAHREN_ERROR_NOT_INITIALIZED;
    }
    if (cm->input_dim != input_dim) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: input_dim mismatch (model=%zu, arg=%zu)\n", cm->input_dim, input_dim);
        #endif
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }

    FILE* f = fopen(weights_path, "rb+");
    if (!f) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Could not open weights file '%s': %s\n", weights_path, strerror(errno));
        #endif
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    /* Training loop */
    for (size_t e = 0; e < epochs; ++e) {
        double epoch_loss = 0.0;
        for (size_t s = 0; s < sample_count; ++s) {
            const float* x = &inputs[s * input_dim];
            int y = labels[s];

            float** layer_outputs = NULL;
            size_t* layer_sizes = (size_t*)calloc(cm->layer_count + 1, sizeof(size_t));
            if (!layer_sizes) { fclose(f); return FAHREN_ERROR_PROCESSING_FAILED; }
            if (forward_sample(cm, f, x, &layer_outputs, layer_sizes) != 0) { fclose(f); return FAHREN_ERROR_PROCESSING_FAILED; }

            /* Loss: -log p_y */
            float py = 1e-9f;
            if (y >= 0 && (size_t)y < layer_sizes[cm->layer_count]) py = layer_outputs[cm->layer_count][y];
            epoch_loss += -logf(py);

            if (backward_update(cm, f, layer_outputs, layer_sizes, y, learning_rate) != 0) { fclose(f); return FAHREN_ERROR_PROCESSING_FAILED; }

            free_layer_outputs(layer_outputs, cm->layer_count);
            free(layer_sizes);
        }
        #if FAHREN_VERBOSE
        fprintf(stdout, "Epoch %zu/%zu - loss: %.6f\n", e + 1, epochs, epoch_loss / (double)sample_count);
        #endif
    }

    fflush(f);
    fclose(f);
    return FAHREN_SUCCESS;
}
