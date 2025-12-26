/* Simple POSIX implementation file for FAHREN.
 * Keep internals local; this file implements the minimal API declared in
 * include/FAHREN/fahren.h. The code below is written to be straightforward
 * and easy for readers to follow. */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#include <fahren/fahren.h>
#include <math.h>

/* RNG helper local to the implementation. Returns float in [-0.5, 0.5]. */
static inline float fahren_random_weight(void) {
#if defined(__APPLE__) || defined(HAVE_ARC4RANDOM)
    return ((float)arc4random() / (float)UINT32_MAX) - 0.5f;
#else
    return (float)(drand48() - 0.5);
#endif
}

/* forward declaration for function defined later in this file */
FAHRENStatus fahren_write_random_weights(FAHREN* cm, const char* path);

FAHRENStatus fahren_init(FAHREN* cm, FAHRENModelType model_type, size_t layer_count, FAHRENLayer* layers) {
    if (!cm || !layers || layer_count == 0) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
    cm->model_type = model_type;
    cm->layer_count = layer_count;
    cm->layers = layers;

    cm->initialized = 1;

    /* write initial random weights & biases for inspection */
    (void)fahren_write_random_weights(cm, "fahren_initial_model.bin");

    return FAHREN_SUCCESS;
}

/* Simple typed allocator to allocate an array of FAHRENLayer and zero it.
 * This avoids callers needing to cast the result of `calloc` in C. */
FAHRENLayer* fahren_alloc_layers(size_t count) {
    if (count == 0) return NULL;
    FAHRENLayer* p = (FAHRENLayer*)calloc(count, sizeof(FAHRENLayer));
    return p;
}

FAHRENStatus fahren_shutdown(FAHREN* cm) {
    if (!cm) return FAHREN_ERROR_INVALID_ARGUMENT;
    if (!cm->initialized) return FAHREN_ERROR_NOT_INITIALIZED;

    /* Free allocated layer array if present */
    if (cm->layers) {
        free(cm->layers);
        cm->layers = NULL;
    }

    /* Reset model state */
    cm->initialized = 0;
    cm->layer_count = 0;
    cm->model_type = 0;

    /* Cleanup transient files created by the library. Preserve model
     * binaries ending in .bin; remove files starting with 'fahren_' that
     * are not .bin. Ignore remove errors. */
    DIR* d = opendir(".");
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            const char* name = ent->d_name;
            if (!name) continue;
            if (strncmp(name, "fahren_", 7) != 0) continue;
            size_t len = strlen(name);
            if (len >= 4 && strcmp(name + len - 4, ".bin") == 0) continue; /* keep binaries */
            (void)unlink(name);
        }
        closedir(d);
    }

    return FAHREN_SUCCESS;
}

/* Note: text-processing helper `fahren_process_data` removed to keep the
 * public API minimal. Examples should implement their own simple text
 * handling when needed. */

/* Compute total number of weights for the model, allocate a float buffer,
 * fill it with random values in [-0.5,0.5] using fahr en_random_weight(), and
 * write the raw floats to `path` as a binary blob. The weight-counting rule
 * is heuristic: for dense and recurrent layers we use previous->density *
 * layer->density; for convolutional layers we multiply that by a 3x3 kernel
 * factor (9). If a layer has no `previous_layer` we treat input dim as 1.
 */
FAHRENStatus fahren_write_random_weights(FAHREN* cm, const char* path) {
    if (!cm || !path) return FAHREN_ERROR_INVALID_ARGUMENT;
    if (!cm->initialized) return FAHREN_ERROR_NOT_INITIALIZED;
    /* Count weights and biases */
    size_t total_weights = 0;
    size_t total_biases = 0;
    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayer* layer = &cm->layers[i];
        size_t in_dim = layer->previous_layer ? (size_t)layer->previous_layer->density : 1;
        size_t out_dim = (size_t)layer->density;
        size_t layer_weights = in_dim * out_dim;
        if (layer->layer_type == FAHREN_LAYER_CONVOLUTIONAL) {
            layer_weights *= 9; /* assume 3x3 kernels */
        }
        if (layer_weights > SIZE_MAX - total_weights) return FAHREN_ERROR_PROCESSING_FAILED;
        total_weights += layer_weights;

        if (out_dim > SIZE_MAX - total_biases) return FAHREN_ERROR_PROCESSING_FAILED;
        total_biases += out_dim; /* one bias per output unit/filter */
    }

    /* Prepare buffers */
    float* weights = NULL;
    float* biases = NULL;
    if (total_weights > 0) {
        weights = (float*)malloc(total_weights * sizeof(float));
        if (!weights) return FAHREN_ERROR_PROCESSING_FAILED;
    }
    if (total_biases > 0) {
        biases = (float*)malloc(total_biases * sizeof(float));
        if (!biases) {
            free(weights);
            return FAHREN_ERROR_PROCESSING_FAILED;
        }
    }

    /* Fill random values */
    size_t widx = 0, bidx = 0;
    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayer* layer = &cm->layers[i];
        size_t in_dim = layer->previous_layer ? (size_t)layer->previous_layer->density : 1;
        size_t out_dim = (size_t)layer->density;
        size_t layer_weights = in_dim * out_dim;
        if (layer->layer_type == FAHREN_LAYER_CONVOLUTIONAL) layer_weights *= 9;
        for (size_t k = 0; k < layer_weights; ++k) {
            weights[widx++] = fahren_random_weight();
        }
        for (size_t k = 0; k < out_dim; ++k) {
            biases[bidx++] = fahren_random_weight();
        }
    }

    /* Write binary blob with a small header: magic, version, counts */
    FILE* f = fopen(path, "wb");
    if (!f) {
        free(weights);
        free(biases);
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    /* header */
    uint32_t magic = 0x4641484E; /* 'FAHN' */
    uint32_t ver_major = FAHREN_VERSION_MAJOR;
    uint32_t ver_minor = FAHREN_VERSION_MINOR;
    uint32_t ver_patch = FAHREN_VERSION_PATCH;
    uint64_t wcount = (uint64_t)total_weights;
    uint64_t bcount = (uint64_t)total_biases;

    if (fwrite(&magic, sizeof(magic), 1, f) != 1) goto io_error;
    if (fwrite(&ver_major, sizeof(ver_major), 1, f) != 1) goto io_error;
    if (fwrite(&ver_minor, sizeof(ver_minor), 1, f) != 1) goto io_error;
    if (fwrite(&ver_patch, sizeof(ver_patch), 1, f) != 1) goto io_error;
    if (fwrite(&wcount, sizeof(wcount), 1, f) != 1) goto io_error;
    if (fwrite(&bcount, sizeof(bcount), 1, f) != 1) goto io_error;

    /* weights then biases */
    if (total_weights > 0) {
        if (fwrite(weights, sizeof(float), total_weights, f) != total_weights) goto io_error;
    }
    if (total_biases > 0) {
        if (fwrite(biases, sizeof(float), total_biases, f) != total_biases) goto io_error;
    }

    fclose(f);
    free(weights);
    free(biases);
    return FAHREN_SUCCESS;

io_error:
    fclose(f);
    free(weights);
    free(biases);
    return FAHREN_ERROR_PROCESSING_FAILED;
}

/* Train a simple linear softmax model via SGD and write to `path`.
 * File format: magic('FAHM'), ver_major, ver_minor, ver_patch, input_dim(uint32), output_dim(uint32),
 * followed by W (float[input_dim*output_dim]) in row-major (input-major), then b (float[output_dim]). */
/* Training helper removed from public API and implementation.
 * Keep model writer above for generating initial random weights. */

/* Prediction helper removed — consumers should load model blobs and implement
 * prediction logic themselves if needed. */