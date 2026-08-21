#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <math.h>

#include <nova/errors.h>
#include "internal.h"

static int write_header(FILE* f, uint32_t input_dim, uint32_t layer_count) {
    NovaFileHeader h;
    h.magic = NOVA_FILE_MAGIC;
    h.version = NOVA_FILE_VERSION;
    h.layer_count = layer_count;
    h.input_dim = input_dim;
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    if (fwrite(&h, sizeof(h), 1, f) != 1) return -1;
    return 0;
}

int nova_read_header(FILE* f, NovaFileHeader* out) {
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    if (fread(out, sizeof(*out), 1, f) != 1) return -1;
    if (out->magic != NOVA_FILE_MAGIC || out->version != NOVA_FILE_VERSION) return -1;
    return 0;
}

void nova_model_add_layer(NOVAModel* model, int layer_type, int activation, ...) {
    if (!model || model->current_layer >= model->layer_count) {
        nova_set_last_error("invalid model or layer count exceeded");
        return;
    }

    size_t idx = model->current_layer;
    va_list args;
    va_start(args, activation);

    int density = 0;
    int param1 = 0;
    int param2 = 0;
    void* sub_model = NULL;

    if (layer_type == NOVA_LAYER_SUBMODEL) {
        sub_model = va_arg(args, void*);
        if (!sub_model) { va_end(args); return; }
    } else if (layer_type == NOVA_LAYER_CONVOLUTIONAL) {
        density = va_arg(args, int);
        param1 = va_arg(args, int);
        param2 = va_arg(args, int);
    } else if (layer_type == NOVA_LAYER_POOLING) {
        param1 = va_arg(args, int);
        param2 = va_arg(args, int);
    } else {
        density = va_arg(args, int);
    }

    va_end(args);

    model->layers[idx].layer_type = layer_type;
    model->layers[idx].activation = activation;
    model->layers[idx].density = density;
    model->layers[idx].sub_model = sub_model;
    model->layers[idx].param1 = param1;
    model->layers[idx].param2 = param2;
    model->layers[idx].input_size = 0;
    model->layers[idx].output_size = (layer_type == NOVA_LAYER_DENSE) ? (size_t)density : 0;
    model->current_layer++;
}

NOVAModel* nova_model_create(int model_type, int layer_count) {
    if (layer_count <= 0 || layer_count > NOVA_MAX_LAYERS) {
        nova_set_last_error("invalid layer count");
        return NULL;
    }

    NOVAModel* model = (NOVAModel*)calloc(1, sizeof(NOVAModel));
    if (!model) {
        nova_set_last_error("memory allocation failed for model");
        return NULL;
    }

    model->layers = (NOVALayer*)calloc((size_t)layer_count, sizeof(NOVALayer));
    if (!model->layers) {
        free(model);
        nova_set_last_error("memory allocation failed for layers");
        return NULL;
    }

    model->model_type = model_type;
    model->layer_count = (size_t)layer_count;
    model->initialized = 1;
    model->finalized = 0;
    model->current_layer = 0;
    model->input_dim = 0;
    model->path = NULL;
    return model;
}

void nova_model_destroy(NOVAModel* model) {
    if (!model) return;
    if (model->layers) {
        free(model->layers);
        model->layers = NULL;
    }
    if (model->path) {
        free(model->path);
        model->path = NULL;
    }
    nova_weights_free_cache(model);
    free(model);
}

NOVA_Status nova_write_binary(NOVAModel* model, const char* path, int input_dim) {
    if (!model || !model->initialized || model->current_layer != model->layer_count)
        return NOVA_ERROR_INVALID_ARGUMENT;
    if (input_dim <= 0) return NOVA_ERROR_INVALID_ARGUMENT;

    FILE* f = fopen(path, "wb+");
    if (!f) {
        char detail[256];
        snprintf(detail, sizeof(detail), "could not open '%s': %s", path, strerror(errno));
        nova_set_last_error(detail);
        return NOVA_ERROR_IO;
    }

    if (write_header(f, (uint32_t)input_dim, (uint32_t)model->layer_count) != 0) {
        fclose(f);
        return NOVA_ERROR_PROCESSING_FAILED;
    }

    size_t in_size = (size_t)input_dim;

    if (fseek(f, (long)sizeof(NovaFileHeader), SEEK_SET) != 0) {
        fclose(f);
        return NOVA_ERROR_PROCESSING_FAILED;
    }

    for (size_t i = 0; i < model->layer_count; ++i) {
        NOVALayer* L = &model->layers[i];
        if (L->layer_type != NOVA_LAYER_DENSE) {
            fclose(f);
            nova_set_last_error("only DENSE layers supported");
            return NOVA_ERROR_UNSUPPORTED;
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
            return NOVA_ERROR_IO;
        }

        size_t wcount = L->input_size * L->output_size;
        float limit = (float)sqrt(6.0f / (float)(L->input_size + L->output_size));

        float* wbuf = (float*)malloc(wcount * sizeof(float));
        if (!wbuf) { fclose(f); return NOVA_ERROR_OUT_OF_MEMORY; }
        for (size_t k = 0; k < wcount; ++k)
            wbuf[k] = nova_rand_uniform(-limit, limit);
        if (fwrite(wbuf, sizeof(float), wcount, f) != wcount) {
            free(wbuf); fclose(f); return NOVA_ERROR_IO;
        }
        free(wbuf);

        float* bbuf = (float*)calloc(L->output_size, sizeof(float));
        if (!bbuf) { fclose(f); return NOVA_ERROR_OUT_OF_MEMORY; }
        if (fwrite(bbuf, sizeof(float), L->output_size, f) != L->output_size) {
            free(bbuf); fclose(f); return NOVA_ERROR_IO;
        }
        free(bbuf);

        in_size = L->output_size;
    }

    fflush(f);
    fclose(f);
    return NOVA_SUCCESS;
}

NOVA_Status nova_model_finalize(NOVAModel* model, const char* path, int input_dim) {
    if (!model || !path) return NOVA_ERROR_INVALID_ARGUMENT;
    NOVA_Status rc = nova_write_binary(model, path, input_dim);
    if (rc != NOVA_SUCCESS) return rc;

    model->finalized = 1;
    model->input_dim = (size_t)input_dim;
    size_t len = strlen(path);
    model->path = (char*)malloc(len + 1);
    if (!model->path) return NOVA_ERROR_OUT_OF_MEMORY;
    memcpy(model->path, path, len + 1);
    return NOVA_SUCCESS;
}
