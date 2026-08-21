#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <nova/errors.h>
#include "internal.h"

static void free_layer_params(NOVALayerParams* lp) {
    if (!lp) return;
    free(lp->weights);
    free(lp->biases);
    free(lp->grad_weights);
    free(lp->grad_biases);
    free(lp->opt_state_w);
    free(lp->opt_state_b);
    memset(lp, 0, sizeof(*lp));
}

static void destroy_cache(NOVAWeightCache* cache, size_t layers_initialized) {
    if (!cache) return;
    for (size_t j = 0; j < layers_initialized; ++j)
        free_layer_params(&cache->layers[j]);
    free(cache->layers);
    free(cache->dirpath);
    free(cache);
}

void nova_weights_free_cache(NOVAModel* model) {
    if (!model || !model->cache) return;
    for (size_t i = 0; i < model->cache->layer_count; ++i)
        free_layer_params(&model->cache->layers[i]);
    free(model->cache->layers);
    free(model->cache->dirpath);
    free(model->cache);
    model->cache = NULL;
}

NOVA_Status nova_weights_load(NOVAModel* model, const char* path) {
    if (!model || !path || !model->finalized)
        return NOVA_ERROR_INVALID_ARGUMENT;

    if (model->cache && model->cache->loaded && model->cache->dirpath &&
        strcmp(model->cache->dirpath, path) == 0)
        return NOVA_SUCCESS;

    nova_weights_free_cache(model);

    FILE* f = fopen(path, "rb");
    if (!f) {
        nova_set_last_error("could not open weights file for read");
        return NOVA_ERROR_IO;
    }

    NovaFileHeader hdr;
    if (nova_read_header(f, &hdr) != 0) {
        fclose(f);
        return NOVA_ERROR_FORMAT;
    }
    if (hdr.layer_count != (uint32_t)model->layer_count ||
        hdr.input_dim != (uint32_t)model->input_dim) {
        fclose(f);
        return NOVA_ERROR_LAYER_MISMATCH;
    }

    NOVAWeightCache* cache = (NOVAWeightCache*)calloc(1, sizeof(*cache));
    if (!cache) { fclose(f); return NOVA_ERROR_OUT_OF_MEMORY; }

    cache->layer_count = model->layer_count;
    cache->layers = (NOVALayerParams*)calloc(cache->layer_count, sizeof(NOVALayerParams));
    if (!cache->layers) { free(cache); fclose(f); return NOVA_ERROR_OUT_OF_MEMORY; }

    size_t path_len = strlen(path);
    cache->dirpath = (char*)malloc(path_len + 1);
    if (!cache->dirpath) { destroy_cache(cache, 0); fclose(f); return NOVA_ERROR_OUT_OF_MEMORY; }
    memcpy(cache->dirpath, path, path_len + 1);

    if (fseek(f, (long)sizeof(NovaFileHeader), SEEK_SET) != 0) {
        fclose(f); destroy_cache(cache, 0); return NOVA_ERROR_IO;
    }

    for (size_t i = 0; i < model->layer_count; ++i) {
        NOVALayer* L = &model->layers[i];
        NOVALayerParams* P = &cache->layers[i];

        uint32_t meta[4];
        if (fread(meta, sizeof(uint32_t), 4, f) != 4) {
            fclose(f); destroy_cache(cache, i); return NOVA_ERROR_IO;
        }

        L->layer_type = (int)meta[0];
        L->activation = (int)meta[1];
        L->input_size = (size_t)meta[2];
        L->output_size = (size_t)meta[3];

        P->weight_count = L->input_size * L->output_size;
        P->bias_count = L->output_size;

        P->weights = (float*)malloc(P->weight_count * sizeof(float));
        P->biases = (float*)malloc(P->bias_count * sizeof(float));
        P->grad_weights = (float*)calloc(P->weight_count, sizeof(float));
        P->grad_biases = (float*)calloc(P->bias_count, sizeof(float));
        if (!P->weights || !P->biases || !P->grad_weights || !P->grad_biases) {
            fclose(f); destroy_cache(cache, i); return NOVA_ERROR_OUT_OF_MEMORY;
        }

        if (fread(P->weights, sizeof(float), P->weight_count, f) != P->weight_count) {
            fclose(f); destroy_cache(cache, i + 1); return NOVA_ERROR_IO;
        }

        if (fread(P->biases, sizeof(float), P->bias_count, f) != P->bias_count) {
            fclose(f); destroy_cache(cache, i + 1); return NOVA_ERROR_IO;
        }
    }

    fclose(f);
    cache->loaded = 1;
    cache->dirty = 0;
    model->cache = cache;
    return NOVA_SUCCESS;
}

NOVA_Status nova_weights_flush(NOVAModel* model, const char* path) {
    if (!model || !path || !model->cache || !model->cache->loaded)
        return NOVA_ERROR_NOT_INITIALIZED;
    if (!model->cache->dirty) return NOVA_SUCCESS;

    FILE* f = fopen(path, "wb");
    if (!f) {
        nova_set_last_error("could not open weights file for write");
        return NOVA_ERROR_IO;
    }

    NovaFileHeader hdr;
    hdr.magic = NOVA_FILE_MAGIC;
    hdr.version = NOVA_FILE_VERSION;
    hdr.layer_count = (uint32_t)model->layer_count;
    hdr.input_dim = (uint32_t)model->input_dim;
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NOVA_ERROR_IO; }

    for (size_t i = 0; i < model->layer_count; ++i) {
        NOVALayer* L = &model->layers[i];
        NOVALayerParams* P = &model->cache->layers[i];

        uint32_t meta[4];
        meta[0] = (uint32_t)L->layer_type;
        meta[1] = (uint32_t)L->activation;
        meta[2] = (uint32_t)L->input_size;
        meta[3] = (uint32_t)L->output_size;
        if (fwrite(meta, sizeof(uint32_t), 4, f) != 4) { fclose(f); return NOVA_ERROR_IO; }
        if (fwrite(P->weights, sizeof(float), P->weight_count, f) != P->weight_count) { fclose(f); return NOVA_ERROR_IO; }
        if (fwrite(P->biases, sizeof(float), P->bias_count, f) != P->bias_count) { fclose(f); return NOVA_ERROR_IO; }
    }

    fclose(f);
    model->cache->dirty = 0;
    return NOVA_SUCCESS;
}
