#ifndef NOVA_MNIST_H
#define NOVA_MNIST_H

#include <stddef.h>
#include <nova/errors.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOVA_MNIST_IMAGE_SIZE 784
#define NOVA_MNIST_NUM_CLASSES 10
#define NOVA_MNIST_IMAGE_ROWS 28
#define NOVA_MNIST_IMAGE_COLS 28

NOVA_Status nova_mnist_load(const char* images_file, const char* labels_file,
                            float** images_out, int** labels_out,
                            size_t* num_samples_out);
void nova_mnist_free(float* images, int* labels);
NOVA_Status nova_mnist_load_train(const char* dir, float** images, int** labels,
                                  size_t* num_out);
NOVA_Status nova_mnist_load_test(const char* dir, float** images, int** labels,
                                 size_t* num_out);
NOVA_Status nova_mnist_batch(const float* images, const int* labels,
                             size_t num_samples, size_t batch_size,
                             size_t batch_idx, const float** batch_images,
                             const int** batch_labels, size_t* batch_count);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_MNIST_H */
