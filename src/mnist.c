#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include <nova/mnist.h>
#include "internal.h"

#define NOVA_MNIST_IMAGE_MAGIC  0x00000803u
#define NOVA_MNIST_LABEL_MAGIC  0x00000801u
#define NOVA_MNIST_IMAGE_SIZE   784
#define NOVA_MNIST_MAX_LABEL    9

typedef struct {
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
} NovaMNISTImageHeader;

typedef struct {
    uint32_t magic;
    uint32_t num_items;
} NovaMNISTLabelHeader;

static uint32_t read_be_u32(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

static float* load_images(const char* path, uint32_t* num_out) {
    if (!path || !num_out) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        nova_set_last_error("cannot open MNIST image file");
        return NULL;
    }

    uint32_t magic = read_be_u32(f);
    uint32_t num_items = read_be_u32(f);
    uint32_t num_rows = read_be_u32(f);
    uint32_t num_cols = read_be_u32(f);

    if (magic != NOVA_MNIST_IMAGE_MAGIC) {
        nova_set_last_error("invalid MNIST image magic number");
        fclose(f); return NULL;
    }
    if (num_rows != 28 || num_cols != 28) {
        nova_set_last_error("MNIST images must be 28x28");
        fclose(f); return NULL;
    }

    size_t total = (size_t)num_items * NOVA_MNIST_IMAGE_SIZE;
    float* images = (float*)malloc(total * sizeof(float));
    if (!images) {
        nova_set_last_error("memory allocation failed for MNIST images");
        fclose(f); return NULL;
    }

    size_t buf_size = 1024 * 1024;
    uint8_t* buf = (uint8_t*)malloc(buf_size);
    if (!buf) {
        free(images); fclose(f);
        nova_set_last_error("failed to allocate I/O buffer");
        return NULL;
    }

    size_t pixels_read = 0;
    while (pixels_read < total) {
        size_t to_read = (total - pixels_read) > buf_size ? buf_size : (total - pixels_read);
        size_t actually_read = fread(buf, 1, to_read, f);
        if (actually_read == 0) {
            nova_set_last_error("premature EOF reading MNIST images");
            free(images); free(buf); fclose(f); return NULL;
        }
        for (size_t i = 0; i < actually_read; i++)
            images[pixels_read + i] = (float)buf[i] / 255.0f;
        pixels_read += actually_read;
    }

    free(buf);
    fclose(f);
    *num_out = num_items;
    return images;
}

static int* load_labels(const char* path, uint32_t* num_out) {
    if (!path || !num_out) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        nova_set_last_error("cannot open MNIST label file");
        return NULL;
    }

    uint32_t magic = read_be_u32(f);
    uint32_t num_items = read_be_u32(f);

    if (magic != NOVA_MNIST_LABEL_MAGIC) {
        nova_set_last_error("invalid MNIST label magic number");
        fclose(f); return NULL;
    }

    int* labels = (int*)malloc(num_items * sizeof(int));
    if (!labels) {
        nova_set_last_error("memory allocation failed for MNIST labels");
        fclose(f); return NULL;
    }

    for (uint32_t i = 0; i < num_items; i++) {
        uint8_t byte;
        if (fread(&byte, 1, 1, f) != 1) {
            nova_set_last_error("failed to read MNIST label");
            free(labels); fclose(f); return NULL;
        }
        if (byte > NOVA_MNIST_MAX_LABEL) {
            nova_set_last_error("invalid MNIST label value");
            free(labels); fclose(f); return NULL;
        }
        labels[i] = (int)byte;
    }

    fclose(f);
    *num_out = num_items;
    return labels;
}

NOVA_Status nova_mnist_load(const char* images_file, const char* labels_file,
                            float** images_out, int** labels_out,
                            size_t* num_samples_out) {
    if (!images_file || !labels_file || !images_out || !labels_out || !num_samples_out)
        return NOVA_ERROR_INVALID_ARGUMENT;

    *images_out = NULL;
    *labels_out = NULL;
    *num_samples_out = 0;

    uint32_t num_images = 0;
    float* images = load_images(images_file, &num_images);
    if (!images) return NOVA_ERROR_PROCESSING_FAILED;

    uint32_t num_labels = 0;
    int* labels = load_labels(labels_file, &num_labels);
    if (!labels) {
        free(images);
        return NOVA_ERROR_PROCESSING_FAILED;
    }

    if (num_images != num_labels) {
        nova_set_last_error("MNIST image/label count mismatch");
        free(images); free(labels);
        return NOVA_ERROR_INVALID_ARGUMENT;
    }

    *images_out = images;
    *labels_out = labels;
    *num_samples_out = (size_t)num_images;
    return NOVA_SUCCESS;
}

void nova_mnist_free(float* images, int* labels) {
    free(images);
    free(labels);
}

NOVA_Status nova_mnist_batch(const float* images, const int* labels,
                             size_t num_samples, size_t batch_size,
                             size_t batch_idx, const float** batch_images,
                             const int** batch_labels, size_t* batch_count) {
    if (!images || !labels || !batch_images || !batch_labels || !batch_count)
        return NOVA_ERROR_INVALID_ARGUMENT;
    if (batch_size == 0) return NOVA_ERROR_INVALID_ARGUMENT;

    size_t start = batch_idx * batch_size;
    if (start >= num_samples) {
        *batch_count = 0;
        *batch_images = NULL;
        *batch_labels = NULL;
        return NOVA_SUCCESS;
    }

    size_t end = start + batch_size;
    if (end > num_samples) end = num_samples;

    *batch_images = &images[start * NOVA_MNIST_IMAGE_SIZE];
    *batch_labels = &labels[start];
    *batch_count = end - start;
    return NOVA_SUCCESS;
}
