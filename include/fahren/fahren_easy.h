/*
 * FAHREN Easy API — short, readable bindings for common workflows.
 */
#ifndef FAHREN_EASY_H
#define FAHREN_EASY_H

#include <fahren/fahren.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FahrenEasyMnistPaths {
    char train_images[512];
    char train_labels[512];
    char test_images[512];
    char test_labels[512];
} FahrenEasyMnistPaths;

/* Resolve standard MNIST IDX filenames under dir. Returns 0 on success. */
int fahren_easy_mnist_paths(const char* dir, FahrenEasyMnistPaths* paths);

/* End marker for fahren_easy_model_dense / fahren_easy_open_dense varargs. */
#define FAHREN_EASY_END (-1)

/*
 * Build a sequential dense model. Varargs: (activation, units) per layer.
 * Last layer should use FAHREN_LAYER_ACTIVATION_SOFTMAX for classifiers.
 * Terminate with FAHREN_EASY_END (not 0 — RELU is 0).
 */
FAHRENModel* fahren_easy_model_dense(int input_dim, const char* weights_path, ...);

/* Open existing weights (same layer layout as model_dense). */
FAHRENModel* fahren_easy_open_dense(int input_dim, const char* weights_path, ...);

/* Train using weights_path on the model (model must be finalized). */
int fahren_easy_train(FAHRENModel* model, const char* weights_path,
                      const float* inputs, const int* labels,
                      size_t sample_count, size_t input_dim, size_t num_classes,
                      size_t epochs, const FAHRENTrainConfig* config);

/* Returns accuracy in [0, 1]. */
int fahren_easy_accuracy(FAHRENModel* model, const char* weights_path,
                        const float* inputs, const int* labels,
                        size_t sample_count, size_t input_dim, float* accuracy_out);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_EASY_H */
