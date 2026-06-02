/*
 * MIT License
 *
 * Copyright (c) 2025 Imran Mukhiddinov <imranmukhiddinov2009@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* ============================================================================
 * FAHREN - Fast and Readable Neural Network Library
 * ============================================================================
 * 
 * A minimal, easy-to-read neural network library implementing:
 * - Sequential model architecture
 * - Dense, Convolutional, Pooling, and Submodel layers
 * - Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
 * - Training via backpropagation with gradient descent
 * - Binary weight file I/O for model persistence
 * 
 * QUICK START:
 * 
 *   1. Create model:
 *      FAHRENModel* model = fahren_create_model(FAHREN_MODEL_SEQUENTIAL, 3);
 * 
 *   2. Add layers:
 *      fahren_add_layer(model, FAHREN_LAYER_DENSE, FAHREN_LAYER_ACTIVATION_RELU, 32);
 *      fahren_add_layer(model, FAHREN_LAYER_DENSE, FAHREN_LAYER_ACTIVATION_RELU, 16);
 *      fahren_add_layer(model, FAHREN_LAYER_DENSE, FAHREN_LAYER_ACTIVATION_SOFTMAX, 2);
 * 
 *   3. Finalize model:
 *      fahren_finalize_model_to_file(model, "weights.bin", input_dim);
 * 
 *   4. Train:
 *      FAHRENTrainConfig cfg = fahren_train_config_default(0.01f);
 *      fahren_train_with_config(model, inputs, num_samples, input_dim,
 *              labels, num_classes, "weights.bin", epochs, &cfg);
 * 
 *   5. Cleanup:
 *      fahren_shutdown(model);
 *
 * Weights file layout: see docs/weights-format.md (magic 'FAHR', version 1).
 */
#ifndef FAHREN_H
#define FAHREN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <fahren/errors.h>
#include <fahren/utils/optimizers.h>

/* Verbosity control: redefine to 1 to enable verbose output (default: quiet) */
#ifndef FAHREN_VERBOSE
#define FAHREN_VERBOSE 0
#endif

/* Version */
#define FAHREN_VERSION_MAJOR                1
#define FAHREN_VERSION_MINOR                0
#define FAHREN_VERSION_PATCH                0

#define FAHREN_MODEL_SEQUENTIAL             0
#define FAHREN_MODEL_LSTM                   1

#define FAHREN_LAYER_DENSE                  0
#define FAHREN_LAYER_CONVOLUTIONAL          1
#define FAHREN_LAYER_POOLING                2
#define FAHREN_LAYER_SUBMODEL               3

#define FAHREN_LAYER_ACTIVATION_RELU        0
#define FAHREN_LAYER_ACTIVATION_SIGMOID     1
#define FAHREN_LAYER_ACTIVATION_TANH        2
#define FAHREN_LAYER_ACTIVATION_SOFTMAX     3

#define FAHREN_DEVICE_CPU                   0
#define FAHREN_DEVICE_CUDA                  1
#define FAHREN_DEVICE_DEFAULT               (-1)

#define FAHREN_WEIGHTS_MAGIC                0x46414852u /* 'FAHR' */
#define FAHREN_WEIGHTS_VERSION              1u

/* Opaque model instance held by library users; keep fields minimal. */
typedef struct FAHRENModel FAHRENModel;

typedef struct FAHRENTrainConfig {
    size_t batch_size;           /* default 32 when 0 */
    float learning_rate;         /* used for SGD when optimizer is NULL */
    FAHRENOptimizer* optimizer;  /* NULL = vanilla SGD */
    int device;                  /* FAHREN_DEVICE_* or FAHREN_DEVICE_DEFAULT (-1) */
} FAHRENTrainConfig;

/* Public API: simple and self-explanatory names. Signatures are intentionally
 * small so users can easily call them from examples. */

/* Initialize a model instance. Pass a model type and layer count, returns a pointer to the model. */
FAHRENModel* fahren_create_model(int model_type, int layer_count);

/* Shutdown and free resources associated with a model. */
void fahren_shutdown(FAHRENModel* cm);

/* Add a layer to the model. */
void fahren_add_layer(FAHRENModel* cm, int layer_type, int activation, ...);
int fahren_finalize_model_to_file(FAHRENModel* cm, const char* filepath, int input_dim);

FAHRENTrainConfig fahren_train_config_default(float learning_rate);
FAHRENTrainConfig fahren_train_config_cuda(float learning_rate);
int fahren_train_resolve_device(const FAHRENTrainConfig* config);
int fahren_train_with_config(FAHRENModel* cm, const float* inputs, size_t sample_count,
                             size_t input_dim, const int* labels, size_t num_classes,
                             const char* weights_path, size_t epochs,
                             const FAHRENTrainConfig* config);

int fahren_train(FAHRENModel* cm, const float* inputs, size_t sample_count, size_t input_dim,
                 const int* labels, size_t num_classes, const char* weights_path,
                 size_t epochs, float learning_rate);

int fahren_set_device(int device);
int fahren_get_device(void);
int fahren_cuda_available(void);

/* Run forward pass and return the argmax class index (requires softmax on the last layer). */
int fahren_predict(FAHRENModel* cm, const char* weights_path, const float* input,
                   size_t input_dim, int* predicted_class);

/* Fraction of samples classified correctly in [0, 1]. */
int fahren_evaluate(FAHRENModel* cm, const char* weights_path,
                    const float* inputs, const int* labels,
                    size_t sample_count, size_t input_dim, float* accuracy_out);

/* ============================================================================
 * MNIST DATASET UTILITIES
 * ============================================================================
 * 
 * Convenience functions for loading and managing MNIST handwritten digit data.
 * Handles IDX-ubyte format files with proper validation and memory efficiency.
 * 
 * Example usage:
 *   float* images = NULL;
 *   int* labels = NULL;
 *   size_t num_samples = 0;
 * 
 *   int rc = fahren_mnist_load_dataset(
 *       "mnist/t10k-images.idx3-ubyte",
 *       "mnist/t10k-labels.idx1-ubyte",
 *       &images, &labels, &num_samples
 *   );
 * 
 *   if (rc == FAHREN_SUCCESS) {
 *       // Use images and labels for training...
 *       fahren_mnist_free_dataset(images, labels);
 *   }
 */

/* Load MNIST dataset (images and corresponding labels)
 * 
 * Parameters:
 *  - images_file: Path to IDX image file (e.g., "t10k-images.idx3-ubyte")
 *  - labels_file: Path to IDX label file (e.g., "t10k-labels.idx1-ubyte")
 *  - images_out: Pointer to float array [num_samples * 784], pixel values in [0, 1]
 *  - labels_out: Pointer to int array [num_samples], label values in [0, 9]
 *  - num_samples_out: Output number of samples loaded
 * 
 * Returns: FAHREN_SUCCESS on success, FAHREN_ERROR_* on failure
 * 
 * Notes:
 *  - Images are automatically normalized from [0, 255] to [0, 1]
 *  - Memory is allocated here; must be freed with fahren_mnist_free_dataset()
 *  - Validates file format and label ranges
 */
int fahren_mnist_load_dataset(const char* images_file, const char* labels_file,
                             float** images_out, int** labels_out, 
                             size_t* num_samples_out);

/* Free MNIST dataset memory
 * 
 * Parameters:
 *  - images: Image array allocated by fahren_mnist_load_dataset()
 *  - labels: Label array allocated by fahren_mnist_load_dataset()
 * 
 * Safe to call with NULL pointers
 */
void fahren_mnist_free_dataset(float* images, int* labels);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_H */
