/*
 * FAHREN Model Initialization and Layer Management
 * 
 * This module handles:
 * - Model creation and configuration
 * - Layer addition and parameter tracking
 * - Model finalization with random weight initialization
 * - Binary file I/O for weight persistence
 * - Memory management and cleanup
 * 
 * The module uses a file-based weight storage approach:
 * 1. Model finalization writes all layer parameters to a binary file
 * 2. Weights are initialized with Xavier/Glorot uniform distribution
 * 3. Training loads and updates weights from this file
 * 4. File format includes header (magic, version, metadata) + per-layer data
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <math.h>

#include "internal.h"

#define FAHREN_FILE_MAGIC 0x46414852u /* 'FAHR' */
#define FAHREN_FILE_VERSION 1u

typedef struct FahrenFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_count;
    uint32_t input_dim;
} FahrenFileHeader;

static int write_header(FILE* f, uint32_t input_dim, uint32_t layer_count) {
    FahrenFileHeader h;
    h.magic = FAHREN_FILE_MAGIC;
    h.version = FAHREN_FILE_VERSION;
    h.layer_count = layer_count;
    h.input_dim = input_dim;
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    if (fwrite(&h, sizeof(h), 1, f) != 1) return -1;
    return 0;
}

static int read_header(FILE* f, FahrenFileHeader* out) {
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    if (fread(out, sizeof(*out), 1, f) != 1) return -1;
    if (out->magic != FAHREN_FILE_MAGIC || out->version != FAHREN_FILE_VERSION) return -1;
    return 0;
}

/* Public API: add layer with activation */
void fahren_add_layer(FAHRENModel* cm, int layer_type, ...) {
    if (!cm || cm->current_layer >= cm->layer_count) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to add layer due to invalid layer count or model pointer\n");
        #endif
        abort();
    }

    size_t idx = cm->current_layer;
    va_list args;
    va_start(args, layer_type);
    
    int activation = va_arg(args, int);

    FAHRENModel* sub_model = NULL;
    int density = 0;
    int param1 = 0;
    int param2 = 0;

    if (layer_type == FAHREN_LAYER_SUBMODEL) {
        sub_model = va_arg(args, FAHRENModel*);
        if (!sub_model) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: SUBMODEL layer requires a valid sub_model pointer\n");
            #endif
            abort();
        }
        density = 0; /* not used */
    } else if (layer_type == FAHREN_LAYER_CONVOLUTIONAL) {
        density = va_arg(args, int);  /* filters */
        param1 = va_arg(args, int);   /* kernel_size */
        param2 = va_arg(args, int);   /* stride */
        if (density <= 0 || param1 <= 0 || param2 <= 0) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Invalid CONV layer parameters\n");
            #endif
            abort();
        }
    } else if (layer_type == FAHREN_LAYER_POOLING) {
        param1 = va_arg(args, int);   /* pool_size */
        param2 = va_arg(args, int);   /* stride */
        if (param1 <= 0 || param2 <= 0) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Invalid POOLING layer parameters\n");
            #endif
            abort();
        }
    } else { /* DENSE */
        density = va_arg(args, int);  /* units */
        if (density <= 0) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Invalid DENSE units\n");
            #endif
            abort();
        }
    }

    va_end(args);

    cm->layers[idx].density = density;
    cm->layers[idx].layer_type = layer_type;
    cm->layers[idx].activation = activation;
    cm->layers[idx].sub_model = sub_model;
    cm->layers[idx].param1 = param1;
    cm->layers[idx].param2 = param2;
    cm->layers[idx].input_size = 0;   /* set during finalize */
    cm->layers[idx].output_size = (layer_type == FAHREN_LAYER_DENSE) ? (size_t)density : 0;
    cm->layers[idx].weights_offset = -1;
    cm->layers[idx].bias_offset = -1;

    cm->current_layer++;
}

FAHRENModel* fahren_create_model(int model_type, int layer_count) {
    if (layer_count <= 0) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to create model due to invalid layer count\n");
        #endif
        abort();
    }

    FAHRENModel* cm = (FAHRENModel*)calloc(1, sizeof(FAHRENModel));
    if (!cm) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Memory allocation failure for model\n");
        #endif
        abort();
    }

    cm->model_type = model_type;
    cm->layer_count = (size_t)layer_count;
    cm->layers = (FAHRENLayer*)calloc((size_t)layer_count, sizeof(FAHRENLayer));
    if (!cm->layers) {
        free(cm);
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Memory allocation failure for layers\n");
        #endif
        abort();
    }

    cm->initialized = 1;
    cm->finalized = 0;
    cm->current_layer = 0;
    cm->input_dim = 0;
    cm->weights_path = NULL;

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Created model with capacity for %d layers\n", layer_count);
    #endif

    return cm;
}

/* Internal helper: write the binary model file with random-initialized weights. */
int _fahren_write_model_binary(FAHRENModel* cm, const char* filepath, int input_dim) {
    if (!cm || !cm->initialized || cm->current_layer != cm->layer_count) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
    if (input_dim <= 0) return FAHREN_ERROR_INVALID_ARGUMENT;

    FILE* f = fopen(filepath, "wb+");
    if (!f) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Could not open weights file '%s': %s\n", filepath, strerror(errno));
        #endif
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    if (write_header(f, (uint32_t)input_dim, (uint32_t)cm->layer_count) != 0) {
        fclose(f);
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    size_t in_size = (size_t)input_dim;

    /* Write per-layer metadata and random weights/biases */
    if (fseek(f, (long)sizeof(FahrenFileHeader), SEEK_SET) != 0) {
        fclose(f);
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayer* L = &cm->layers[i];
        if (L->layer_type != FAHREN_LAYER_DENSE) {
            #if FAHREN_VERBOSE
            fprintf(stderr, "FAHREN ERROR: Only DENSE layers are supported in this build for finalize/train.\n");
            #endif
            fclose(f);
            return FAHREN_ERROR_INVALID_ARGUMENT;
        }

        L->input_size = in_size;
        L->output_size = (size_t)L->density;

        uint32_t meta[4];
        meta[0] = (uint32_t)L->layer_type;
        meta[1] = (uint32_t)L->activation;
        meta[2] = (uint32_t)L->input_size;
        meta[3] = (uint32_t)L->output_size;
        if (fwrite(meta, sizeof(uint32_t), 4, f) != 4) {
            fclose(f);
            return FAHREN_ERROR_PROCESSING_FAILED;
        }

        /* Record offsets */
        L->weights_offset = ftell(f);
        size_t wcount = L->input_size * L->output_size;

        /* Xavier/Glorot uniform */
        float limit = (float)sqrt(6.0f / (float)(L->input_size + L->output_size));

        /* Allocate a small buffer and fill */
        float* wbuf = (float*)malloc(wcount * sizeof(float));
        if (!wbuf) { fclose(f); return FAHREN_ERROR_PROCESSING_FAILED; }
        for (size_t k = 0; k < wcount; ++k) {
            wbuf[k] = fahren_rand_uniform(-limit, limit);
        }
        if (fwrite(wbuf, sizeof(float), wcount, f) != wcount) {
            free(wbuf);
            fclose(f);
            return FAHREN_ERROR_PROCESSING_FAILED;
        }
        free(wbuf);

        L->bias_offset = ftell(f);
        float* bbuf = (float*)calloc(L->output_size, sizeof(float));
        if (!bbuf) { fclose(f); return FAHREN_ERROR_PROCESSING_FAILED; }
        if (fwrite(bbuf, sizeof(float), L->output_size, f) != L->output_size) {
            free(bbuf);
            fclose(f);
            return FAHREN_ERROR_PROCESSING_FAILED;
        }
        free(bbuf);

        in_size = L->output_size;
    }

    fflush(f);
    fclose(f);
    return FAHREN_SUCCESS;
}

int fahren_finalize_model_to_file(FAHRENModel* cm, const char* filepath, int input_dim) {
    if (!cm || !filepath) return FAHREN_ERROR_INVALID_ARGUMENT;
    int rc = _fahren_write_model_binary(cm, filepath, input_dim);
    if (rc != FAHREN_SUCCESS) return rc;

    cm->finalized = 1;
    cm->input_dim = (size_t)input_dim;
    size_t len = strlen(filepath);
    cm->weights_path = (char*)malloc(len + 1);
    if (!cm->weights_path) return FAHREN_ERROR_PROCESSING_FAILED;
    memcpy(cm->weights_path, filepath, len + 1);

    #if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Finalized model to '%s' (input_dim=%d, layers=%zu)\n", filepath, input_dim, cm->layer_count);
    #endif

    return FAHREN_SUCCESS;
}

void fahren_shutdown(FAHRENModel* cm) {
    if (!cm) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to shutdown model due to invalid model pointer\n");
        #endif
        abort();
    }
    if (!cm->initialized) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Model not initialized\n");
        #endif
        abort();
    }

    if (cm->layers) {
        free(cm->layers);
        cm->layers = NULL;
    }
    if (cm->weights_path) {
        free(cm->weights_path);
        cm->weights_path = NULL;
    }

    free(cm);
}
