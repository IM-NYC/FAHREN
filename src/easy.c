#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <fahren/fahren_easy.h>
#include <fahren/architectures/sequential.h>
#include "internal.h"

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int join_path(char* out, size_t out_len, const char* dir, const char* name) {
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    int need_slash = (dlen > 0 && dir[dlen - 1] != '/' && dir[dlen - 1] != '\\');
    size_t total = dlen + nlen + (need_slash ? 1 : 0) + 1;
    if (total > out_len) return -1;
    snprintf(out, out_len, "%s%s%s", dir, need_slash ? "/" : "", name);
    return 0;
}

static int pick_path(char* out, size_t out_len, const char* dir, const char* a, const char* b) {
    char path[512];
    if (join_path(path, sizeof(path), dir, a) == 0 && file_exists(path)) {
        snprintf(out, out_len, "%s", path);
        return 0;
    }
    if (join_path(path, sizeof(path), dir, b) == 0 && file_exists(path)) {
        snprintf(out, out_len, "%s", path);
        return 0;
    }
    return -1;
}

int fahren_easy_mnist_paths(const char* dir, FahrenEasyMnistPaths* paths) {
    if (!dir || !paths) return FAHREN_ERROR_INVALID_ARGUMENT;
    memset(paths, 0, sizeof(*paths));
    if (pick_path(paths->train_images, sizeof(paths->train_images), dir,
                  "train-images.idx3-ubyte", "train-images-idx3-ubyte") != 0) {
        return FAHREN_ERROR_IO;
    }
    if (pick_path(paths->train_labels, sizeof(paths->train_labels), dir,
                  "train-labels.idx1-ubyte", "train-labels-idx1-ubyte") != 0) {
        return FAHREN_ERROR_IO;
    }
    if (pick_path(paths->test_images, sizeof(paths->test_images), dir,
                  "t10k-images.idx3-ubyte", "test-images.idx3-ubyte") != 0) {
        return FAHREN_ERROR_IO;
    }
    if (pick_path(paths->test_labels, sizeof(paths->test_labels), dir,
                  "t10k-labels.idx1-ubyte", "test-labels.idx1-ubyte") != 0) {
        return FAHREN_ERROR_IO;
    }
    return FAHREN_SUCCESS;
}

#define FAHREN_EASY_DENSE_MAX_LAYERS 32

#define FAHREN_EASY_PARSE_DENSE_LAYERS(ap_var) \
    int acts[FAHREN_EASY_DENSE_MAX_LAYERS]; \
    int units[FAHREN_EASY_DENSE_MAX_LAYERS]; \
    size_t layer_count = 0; \
    for (;;) { \
        int act = va_arg((ap_var), int); \
        if (act == FAHREN_EASY_END) break; \
        if (layer_count >= FAHREN_EASY_DENSE_MAX_LAYERS) return NULL; \
        acts[layer_count] = act; \
        units[layer_count] = va_arg((ap_var), int); \
        ++layer_count; \
    } \
    if (layer_count == 0) return NULL; \
    FAHRENModel* model = fahren_create_model(FAHREN_MODEL_SEQUENTIAL, (int)layer_count); \
    if (!model) return NULL; \
    for (size_t i = 0; i < layer_count; ++i) { \
        fahren_add_layer(model, FAHREN_LAYER_DENSE, acts[i], units[i]); \
    }

FAHRENModel* fahren_easy_model_dense(int input_dim, const char* weights_path, ...) {
    if (!weights_path || input_dim <= 0) return NULL;

    va_list ap;
    va_start(ap, weights_path);
    FAHREN_EASY_PARSE_DENSE_LAYERS(ap);
    va_end(ap);

    if (fahren_finalize_model_to_file(model, weights_path, input_dim) != FAHREN_SUCCESS) {
        fahren_shutdown(model);
        return NULL;
    }
    return model;
}

FAHRENModel* fahren_easy_open_dense(int input_dim, const char* weights_path, ...) {
    if (!weights_path || input_dim <= 0) return NULL;

    va_list ap;
    va_start(ap, weights_path);
    FAHREN_EASY_PARSE_DENSE_LAYERS(ap);
    va_end(ap);

    model->input_dim = (size_t)input_dim;
    model->finalized = 1;
    size_t len = strlen(weights_path);
    model->weights_path = (char*)malloc(len + 1);
    if (!model->weights_path) {
        fahren_shutdown(model);
        return NULL;
    }
    memcpy(model->weights_path, weights_path, len + 1);

    if (fahren_weights_load(model, weights_path) != FAHREN_SUCCESS) {
        fahren_shutdown(model);
        return NULL;
    }
    return model;
}

int fahren_easy_train(FAHRENModel* model, const char* weights_path,
                      const float* inputs, const int* labels,
                      size_t sample_count, size_t input_dim, size_t num_classes,
                      size_t epochs, const FAHRENTrainConfig* config) {
    if (!model || !weights_path || !inputs || !labels || !config) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
    return fahren_train_with_config(model, inputs, sample_count, input_dim, labels,
                                    num_classes, weights_path, epochs, config);
}

int fahren_easy_accuracy(FAHRENModel* model, const char* weights_path,
                         const float* inputs, const int* labels,
                         size_t sample_count, size_t input_dim, float* accuracy_out) {
    return fahren_evaluate(model, weights_path, inputs, labels, sample_count, input_dim,
                           accuracy_out);
}

FAHRENModel* fahren_create_sequential_model(int layer_count) {
    return fahren_create_model(FAHREN_MODEL_SEQUENTIAL, layer_count);
}

void fahren_add_dense_layer(FAHRENModel* cm, int activation, int units) {
    fahren_add_layer(cm, FAHREN_LAYER_DENSE, activation, units);
}

void fahren_add_conv_layer(FAHRENModel* cm, int activation, int filters, int kernel_size, int stride) {
    fahren_add_layer(cm, FAHREN_LAYER_CONVOLUTIONAL, activation, filters, kernel_size, stride);
}

void fahren_add_pooling_layer(FAHRENModel* cm, int pool_size, int stride) {
    fahren_add_layer(cm, FAHREN_LAYER_POOLING, FAHREN_LAYER_ACTIVATION_RELU, pool_size, stride);
}

int fahren_train_sequential(FAHRENModel* cm, const float* inputs, size_t sample_count,
                            size_t input_dim, const int* labels, size_t num_classes,
                            const char* weights_path, size_t epochs, float learning_rate) {
    return fahren_train(cm, inputs, sample_count, input_dim, labels, num_classes,
                        weights_path, epochs, learning_rate);
}

int fahren_predict_sequential(FAHRENModel* cm, const char* weights_path,
                              const float* input, size_t input_dim, float* output) {
    int pred = 0;
    if (!cm || !weights_path || !input || !output) return FAHREN_ERROR_INVALID_ARGUMENT;
    int rc = fahren_predict(cm, weights_path, input, input_dim, &pred);
    if (rc != FAHREN_SUCCESS) return rc;
    *output = (float)pred;
    return FAHREN_SUCCESS;
}
