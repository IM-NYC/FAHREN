#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <fahren/errors.h>
#include "internal.h"

static int read_header(FILE* f, FahrenFileHeader* out) {
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    if (fread(out, sizeof(*out), 1, f) != 1) return -1;
    if (out->magic != FAHREN_FILE_MAGIC || out->version != FAHREN_FILE_VERSION) return -1;
    return 0;
}

static void free_layer_params(FAHRENLayerParams* lp) {
    if (!lp) return;
    free(lp->weights);
    free(lp->biases);
    free(lp->grad_weights);
    free(lp->grad_biases);
    fahren_optimizer_state_free(lp->opt_state_w);
    fahren_optimizer_state_free(lp->opt_state_b);
    memset(lp, 0, sizeof(*lp));
}

static void destroy_weight_cache(FAHRENWeightCache* cache, size_t layers_initialized) {
    if (!cache) return;
    for (size_t j = 0; j < layers_initialized; ++j) {
        free_layer_params(&cache->layers[j]);
    }
    free(cache->layers);
    free(cache->filepath);
    free(cache);
}

void fahren_weights_free_cache(struct FAHRENModel* cm) {
    if (!cm || !cm->cache) return;
    for (size_t i = 0; i < cm->cache->layer_count; ++i) {
        free_layer_params(&cm->cache->layers[i]);
    }
    free(cm->cache->layers);
    free(cm->cache->filepath);
    free(cm->cache);
    cm->cache = NULL;
}

int fahren_weights_load(struct FAHRENModel* cm, const char* filepath) {
    if (!cm || !filepath || !cm->finalized) return FAHREN_ERROR_INVALID_ARGUMENT;

    if (cm->cache && cm->cache->loaded && cm->cache->filepath &&
        strcmp(cm->cache->filepath, filepath) == 0) {
        return FAHREN_SUCCESS;
    }

    fahren_weights_free_cache(cm);

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fahren_set_last_error("could not open weights file for read");
        return FAHREN_ERROR_IO;
    }

    FahrenFileHeader hdr;
    if (read_header(f, &hdr) != 0) {
        fclose(f);
        return FAHREN_ERROR_FORMAT;
    }
    if (hdr.layer_count != (uint32_t)cm->layer_count || hdr.input_dim != (uint32_t)cm->input_dim) {
        fclose(f);
        return FAHREN_ERROR_LAYER_MISMATCH;
    }

    FAHRENWeightCache* cache = (FAHRENWeightCache*)calloc(1, sizeof(*cache));
    if (!cache) { fclose(f); return FAHREN_ERROR_OUT_OF_MEMORY; }

    cache->layer_count = cm->layer_count;
    cache->layers = (FAHRENLayerParams*)calloc(cache->layer_count, sizeof(FAHRENLayerParams));
    if (!cache->layers) {
        free(cache);
        fclose(f);
        return FAHREN_ERROR_OUT_OF_MEMORY;
    }

    size_t path_len = strlen(filepath);
    cache->filepath = (char*)malloc(path_len + 1);
    if (!cache->filepath) {
        destroy_weight_cache(cache, 0);
        fclose(f);
        return FAHREN_ERROR_OUT_OF_MEMORY;
    }
    memcpy(cache->filepath, filepath, path_len + 1);

    if (fseek(f, (long)sizeof(FahrenFileHeader), SEEK_SET) != 0) {
        fclose(f);
        destroy_weight_cache(cache, 0);
        return FAHREN_ERROR_IO;
    }

    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayer* L = &cm->layers[i];
        FAHRENLayerParams* P = &cache->layers[i];

        uint32_t meta[4];
        if (fread(meta, sizeof(uint32_t), 4, f) != 4) {
            fclose(f);
            destroy_weight_cache(cache, i);
            return FAHREN_ERROR_IO;
        }

        L->layer_type = (int)meta[0];
        L->activation = (int)meta[1];
        L->input_size = (size_t)meta[2];
        L->output_size = (size_t)meta[3];

        P->weight_count = L->input_size * L->output_size;
        P->bias_count = L->output_size;
        L->weights_offset = ftell(f);

        P->weights = (float*)malloc(P->weight_count * sizeof(float));
        P->biases = (float*)malloc(P->bias_count * sizeof(float));
        P->grad_weights = (float*)calloc(P->weight_count, sizeof(float));
        P->grad_biases = (float*)calloc(P->bias_count, sizeof(float));
        if (!P->weights || !P->biases || !P->grad_weights || !P->grad_biases) {
            fclose(f);
            destroy_weight_cache(cache, i);
            return FAHREN_ERROR_OUT_OF_MEMORY;
        }

        if (fread(P->weights, sizeof(float), P->weight_count, f) != P->weight_count) {
            fclose(f);
            destroy_weight_cache(cache, i + 1);
            return FAHREN_ERROR_IO;
        }

        L->bias_offset = ftell(f);
        if (fread(P->biases, sizeof(float), P->bias_count, f) != P->bias_count) {
            fclose(f);
            destroy_weight_cache(cache, i + 1);
            return FAHREN_ERROR_IO;
        }
    }

    fclose(f);
    cache->loaded = 1;
    cache->dirty = 0;
    cm->cache = cache;
    return FAHREN_SUCCESS;
}

int fahren_weights_flush(struct FAHRENModel* cm, const char* filepath) {
    if (!cm || !filepath || !cm->cache || !cm->cache->loaded) {
        return FAHREN_ERROR_NOT_INITIALIZED;
    }
    if (!cm->cache->dirty) return FAHREN_SUCCESS;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        fahren_set_last_error("could not open weights file for write");
        return FAHREN_ERROR_IO;
    }

    FahrenFileHeader hdr;
    hdr.magic = FAHREN_FILE_MAGIC;
    hdr.version = FAHREN_FILE_VERSION;
    hdr.layer_count = (uint32_t)cm->layer_count;
    hdr.input_dim = (uint32_t)cm->input_dim;
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return FAHREN_ERROR_IO;
    }

    for (size_t i = 0; i < cm->layer_count; ++i) {
        FAHRENLayer* L = &cm->layers[i];
        FAHRENLayerParams* P = &cm->cache->layers[i];

        uint32_t meta[4];
        meta[0] = (uint32_t)L->layer_type;
        meta[1] = (uint32_t)L->activation;
        meta[2] = (uint32_t)L->input_size;
        meta[3] = (uint32_t)L->output_size;
        if (fwrite(meta, sizeof(uint32_t), 4, f) != 4) {
            fclose(f);
            return FAHREN_ERROR_IO;
        }
        if (fwrite(P->weights, sizeof(float), P->weight_count, f) != P->weight_count) {
            fclose(f);
            return FAHREN_ERROR_IO;
        }
        if (fwrite(P->biases, sizeof(float), P->bias_count, f) != P->bias_count) {
            fclose(f);
            return FAHREN_ERROR_IO;
        }
    }

    fclose(f);
    cm->cache->dirty = 0;
    return FAHREN_SUCCESS;
}
