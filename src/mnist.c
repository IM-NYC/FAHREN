/*
 * FAHREN MNIST Data Loading Module
 * 
 * Provides efficient MNIST dataset loading with proper label matching.
 * Uses streaming and buffering for memory efficiency.
 * 
 * Supports:
 * - IDX image format (t10k-images.idx3-ubyte, train-images.idx3-ubyte)
 * - IDX label format (t10k-labels.idx1-ubyte, train-labels.idx1-ubyte)
 * - Pixel normalization (0-255 → 0-1)
 * - Efficient memory management with pre-allocated buffers
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include "internal.h"

/* ============================================================================
 * MNIST DATASET STRUCTURES
 * ============================================================================ */

#define FAHREN_MNIST_IMAGE_SIZE (28 * 28)
#define FAHREN_MNIST_MAX_LABEL 9
#define FAHREN_MNIST_NUM_CLASSES 10

/* IDX file header format (big-endian) */
typedef struct {
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
} FAHRENMNISTImageHeader;

typedef struct {
    uint32_t magic;
    uint32_t num_items;
} FAHRENMNISTLabelHeader;

/* MNIST dataset container */
typedef struct {
    float* images;           /* Flattened images: [num_samples * 784] */
    int* labels;             /* Labels: [num_samples] */
    size_t num_samples;      /* Total number of samples */
    char* image_buffer;      /* Temporary buffer for file I/O */
    size_t buffer_size;
} FAHRENMNISTDataset;

/* ============================================================================
 * BYTE-ORDER UTILITIES (Big-Endian for IDX Format)
 * ============================================================================ */

static uint32_t _fahren_read_big_endian_u32(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | 
           ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

/* ============================================================================
 * MNIST IMAGE LOADING (With Efficient Buffering)
 * ============================================================================ */

static float* _fahren_load_mnist_images_buffered(const char* filepath, 
                                                 uint32_t* num_samples_out) {
    if (!filepath || !num_samples_out) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Invalid arguments to MNIST image loader\n");
        #endif
        return NULL;
    }

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Cannot open MNIST image file: %s (%s)\n", 
                filepath, strerror(errno));
        #endif
        return NULL;
    }

    /* Read and validate header */
    FAHRENMNISTImageHeader header;
    header.magic = _fahren_read_big_endian_u32(f);
    header.num_items = _fahren_read_big_endian_u32(f);
    header.num_rows = _fahren_read_big_endian_u32(f);
    header.num_cols = _fahren_read_big_endian_u32(f);

    /* Validate magic number for images (0x00000803) */
    if (header.magic != 0x00000803) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Invalid MNIST image magic: 0x%08x (expected 0x00000803)\n", 
                header.magic);
        #endif
        fclose(f);
        return NULL;
    }

    /* Validate dimensions (must be 28x28 for standard MNIST) */
    if (header.num_rows != 28 || header.num_cols != 28) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: MNIST images must be 28x28, got %ux%u\n",
                header.num_rows, header.num_cols);
        #endif
        fclose(f);
        return NULL;
    }

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Loading %u MNIST images (%ux%u pixels)\n",
            header.num_items, header.num_rows, header.num_cols);
    #endif

    /* Allocate output array (all pixels as floats) */
    size_t total_pixels = (size_t)header.num_items * 784;
    float* images = (float*)malloc(total_pixels * sizeof(float));
    if (!images) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Memory allocation failed for %zu images\n", 
                total_pixels / 784);
        #endif
        fclose(f);
        return NULL;
    }

    /* Use buffered reading for efficiency (1 MB buffer) */
    size_t buffer_size = 1024 * 1024;  /* 1 MB */
    uint8_t* read_buffer = (uint8_t*)malloc(buffer_size);
    if (!read_buffer) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to allocate I/O buffer\n");
        #endif
        free(images);
        fclose(f);
        return NULL;
    }

    /* Read all pixels and normalize */
    size_t pixels_read = 0;
    while (pixels_read < total_pixels) {
        size_t to_read = (total_pixels - pixels_read) > buffer_size ? 
                        buffer_size : (total_pixels - pixels_read);
        
        size_t actually_read = fread(read_buffer, 1, to_read, f);
        if (actually_read == 0) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Premature EOF reading MNIST images\n");
            #endif
            free(images);
            free(read_buffer);
            fclose(f);
            return NULL;
        }

        /* Normalize pixels: [0, 255] → [0, 1] */
        for (size_t i = 0; i < actually_read; i++) {
            images[pixels_read + i] = (float)read_buffer[i] / 255.0f;
        }

        pixels_read += actually_read;
    }

    free(read_buffer);
    fclose(f);

    *num_samples_out = header.num_items;

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Successfully loaded %u images\n", header.num_items);
    #endif

    return images;
}

/* ============================================================================
 * MNIST LABEL LOADING
 * ============================================================================ */

static int* _fahren_load_mnist_labels(const char* filepath, uint32_t* num_labels_out) {
    if (!filepath || !num_labels_out) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Invalid arguments to MNIST label loader\n");
        #endif
        return NULL;
    }

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Cannot open MNIST label file: %s (%s)\n", 
                filepath, strerror(errno));
        #endif
        return NULL;
    }

    /* Read and validate header */
    FAHRENMNISTLabelHeader header;
    header.magic = _fahren_read_big_endian_u32(f);
    header.num_items = _fahren_read_big_endian_u32(f);

    /* Validate magic number for labels (0x00000801) */
    if (header.magic != 0x00000801) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Invalid MNIST label magic: 0x%08x (expected 0x00000801)\n",
                header.magic);
        #endif
        fclose(f);
        return NULL;
    }

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Loading %u MNIST labels\n", header.num_items);
    #endif

    /* Allocate output array */
    int* labels = (int*)malloc(header.num_items * sizeof(int));
    if (!labels) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Memory allocation failed for %u labels\n",
                header.num_items);
        #endif
        fclose(f);
        return NULL;
    }

    /* Read labels (one byte per label) */
    for (uint32_t i = 0; i < header.num_items; i++) {
        uint8_t label_byte;
        if (fread(&label_byte, 1, 1, f) != 1) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Failed to read label %u\n", i);
            #endif
            free(labels);
            fclose(f);
            return NULL;
        }

        /* Validate label is in range [0, 9] */
        if (label_byte > FAHREN_MNIST_MAX_LABEL) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Invalid label value %u at index %u (expected 0-9)\n",
                    label_byte, i);
            #endif
            free(labels);
            fclose(f);
            return NULL;
        }

        labels[i] = (int)label_byte;
    }

    fclose(f);
    *num_labels_out = header.num_items;

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Successfully loaded %u labels\n", header.num_items);
    #endif

    return labels;
}

/* ============================================================================
 * PUBLIC API: Load Complete MNIST Dataset
 * ============================================================================ */

int fahren_mnist_load_dataset(const char* images_file, const char* labels_file,
                             float** images_out, int** labels_out, size_t* num_samples_out) {
    if (!images_file || !labels_file || !images_out || !labels_out || !num_samples_out) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }

    /* Load images */
    uint32_t num_images = 0;
    float* images = _fahren_load_mnist_images_buffered(images_file, &num_images);
    if (!images) {
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    /* Load labels */
    uint32_t num_labels = 0;
    int* labels = _fahren_load_mnist_labels(labels_file, &num_labels);
    if (!labels) {
        free(images);
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    /* Validate counts match */
    if (num_images != num_labels) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Image count (%u) != Label count (%u)\n",
                num_images, num_labels);
        #endif
        free(images);
        free(labels);
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }

    *images_out = images;
    *labels_out = labels;
    *num_samples_out = (size_t)num_images;

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Loaded MNIST dataset with %zu samples\n", *num_samples_out);
    #endif

    return FAHREN_SUCCESS;
}

/* ============================================================================
 * UTILITY: Free MNIST Dataset Memory
 * ============================================================================ */

void fahren_mnist_free_dataset(float* images, int* labels) {
    if (images) free(images);
    if (labels) free(labels);
}
