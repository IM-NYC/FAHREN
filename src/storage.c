#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

#include <nova/errors.h>
#include "internal.h"

#ifdef _WIN32
#include <direct.h>
#define mkdir_(p) _mkdir(p)
#define strdup_(p) _strdup(p)
#else
#include <sys/stat.h>
#define mkdir_(p) mkdir(p, 0755)
#define strdup_(p) strdup(p)
#endif

void nova_path_native(char* path) {
    if (!path) return;
#if defined(_WIN32)
    for (char* p = path; *p; p++)
        if (*p == '/') *p = '\\';
#else
    (void)path;
#endif
}

NOVA_Status nova_path_join(char* buf, size_t sz, const char* dir, const char* file) {
    if (!buf || !sz || !dir || !file) return NOVA_ERROR_INVALID_ARGUMENT;
    char d[1024];
    snprintf(d, sizeof(d), "%s", dir);
    nova_path_native(d);
    int n = snprintf(buf, sz, "%s" NOVA_PATH_SEP_STR "%s", d, file);
    if (n < 0 || (size_t)n >= sz) return NOVA_ERROR_BUFFER_TOO_SMALL;
    return NOVA_SUCCESS;
}

static int ensure_dir(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
            mkdir_(tmp);
            *p = NOVA_PATH_SEP_CHAR;
        }
    }
    mkdir_(tmp);
    return 0;
}

NOVA_Status nova_save(NOVAModel* model, const char* dirpath) {
    if (!model || !dirpath || !model->finalized)
        return NOVA_ERROR_INVALID_ARGUMENT;

    ensure_dir(dirpath);

    char arch_path[1024], weight_path[1024], bias_path[1024];
    char meta_path[1024], hash_path[1024], train_path[1024];

    nova_path_join(arch_path, sizeof(arch_path), dirpath, "architecture.nova");
    nova_path_join(weight_path, sizeof(weight_path), dirpath, "weights.nova");
    nova_path_join(bias_path, sizeof(bias_path), dirpath, "biases.nova");
    nova_path_join(meta_path, sizeof(meta_path), dirpath, "metadata.nova");
    nova_path_join(hash_path, sizeof(hash_path), dirpath, "checksum.sha256");
    nova_path_join(train_path, sizeof(train_path), dirpath, "training.nova");

    /* Architecture file */
    FILE* f = fopen(arch_path, "wb");
    if (!f) return NOVA_ERROR_IO;

    NovaFileHeader hdr;
    hdr.magic = NOVA_FILE_MAGIC;
    hdr.version = NOVA_FILE_VERSION;
    hdr.layer_count = (uint32_t)model->layer_count;
    hdr.input_dim = (uint32_t)model->input_dim;
    fwrite(&hdr, sizeof(hdr), 1, f);

    for (size_t i = 0; i < model->layer_count; i++) {
        NOVALayer* L = &model->layers[i];
        uint32_t meta[6];
        meta[0] = (uint32_t)L->layer_type;
        meta[1] = (uint32_t)L->activation;
        meta[2] = (uint32_t)L->input_size;
        meta[3] = (uint32_t)L->output_size;
        meta[4] = (uint32_t)L->density;
        meta[5] = (uint32_t)L->param1;
        fwrite(meta, sizeof(uint32_t), 6, f);
    }
    fclose(f);

    /* Weights file */
    f = fopen(weight_path, "wb");
    if (!f) return NOVA_ERROR_IO;
    for (size_t i = 0; i < model->layer_count; i++) {
        NOVALayerParams* P = &model->cache->layers[i];
        uint32_t count = (uint32_t)P->weight_count;
        fwrite(&count, sizeof(count), 1, f);
        fwrite(P->weights, sizeof(float), P->weight_count, f);
    }
    fclose(f);

    /* Biases file */
    f = fopen(bias_path, "wb");
    if (!f) return NOVA_ERROR_IO;
    for (size_t i = 0; i < model->layer_count; i++) {
        NOVALayerParams* P = &model->cache->layers[i];
        uint32_t count = (uint32_t)P->bias_count;
        fwrite(&count, sizeof(count), 1, f);
        fwrite(P->biases, sizeof(float), P->bias_count, f);
    }
    fclose(f);

    /* Metadata file */
    f = fopen(meta_path, "w");
    if (!f) return NOVA_ERROR_IO;
    fprintf(f, "framework=Novaflow\n");
    fprintf(f, "version=%d.%d.%d\n", NOVA_VERSION_MAJOR, NOVA_VERSION_MINOR, NOVA_VERSION_PATCH);
    fprintf(f, "model_type=%d\n", model->model_type);
    fprintf(f, "layer_count=%zu\n", model->layer_count);
    fprintf(f, "input_dim=%zu\n", model->input_dim);
    fprintf(f, "precision=fp32\n");
    fclose(f);

    /* Training file */
    f = fopen(train_path, "w");
    if (!f) return NOVA_ERROR_IO;
    fprintf(f, "optimizer=sgd\n");
    fprintf(f, "epochs=0\n");
    fclose(f);

    /* Checksum file */
    unsigned char hash[32];
    char hex[65];
    f = fopen(hash_path, "w");
    if (!f) return NOVA_ERROR_IO;

    const char* files[] = {"architecture.nova", "weights.nova", "biases.nova",
                           "metadata.nova", "training.nova", NULL};
    for (int i = 0; files[i]; i++) {
        char fp[1024];
        nova_path_join(fp, sizeof(fp), dirpath, files[i]);
        if (nova_hash_file(fp, hash) == NOVA_SUCCESS) {
            nova_hash_to_hex(hash, hex);
            fprintf(f, "%s  %s\n", hex, files[i]);
        }
    }
    fclose(f);

    return NOVA_SUCCESS;
}

NOVA_Status nova_load(NOVAModel* model, const char* dirpath) {
    if (!model || !dirpath) return NOVA_ERROR_INVALID_ARGUMENT;

    char arch_path[1024];
    nova_path_join(arch_path, sizeof(arch_path), dirpath, "architecture.nova");
    FILE* f = fopen(arch_path, "rb");
    if (!f) {
        nova_set_last_error("architecture.nova not found");
        return NOVA_ERROR_INCOMPLETE_MODEL;
    }

    NovaFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f); return NOVA_ERROR_FORMAT;
    }
    if (hdr.magic != NOVA_FILE_MAGIC) {
        fclose(f); return NOVA_ERROR_FORMAT;
    }

    model->layer_count = (size_t)hdr.layer_count;
    model->input_dim = (size_t)hdr.input_dim;
    model->finalized = 1;

    model->layers = (NOVALayer*)calloc(model->layer_count, sizeof(NOVALayer));
    if (!model->layers) { fclose(f); return NOVA_ERROR_OUT_OF_MEMORY; }

    for (size_t i = 0; i < model->layer_count; i++) {
        NOVALayer* L = &model->layers[i];
        uint32_t meta[6];
        if (fread(meta, sizeof(uint32_t), 6, f) != 6) {
            fclose(f); return NOVA_ERROR_FORMAT;
        }
        L->layer_type = (int)meta[0];
        L->activation = (int)meta[1];
        L->input_size = (size_t)meta[2];
        L->output_size = (size_t)meta[3];
        L->density = (int)meta[4];
        L->param1 = (int)meta[5];
    }
    fclose(f);
    model->current_layer = model->layer_count;

    /* Load weights and biases into cache */
    NOVAWeightCache* cache = (NOVAWeightCache*)calloc(1, sizeof(*cache));
    if (!cache) return NOVA_ERROR_OUT_OF_MEMORY;
    cache->layer_count = model->layer_count;
    cache->layers = (NOVALayerParams*)calloc(cache->layer_count, sizeof(NOVALayerParams));
    if (!cache->layers) { free(cache); return NOVA_ERROR_OUT_OF_MEMORY; }

    cache->dirpath = strdup_(dirpath);
    cache->loaded = 1;
    cache->dirty = 0;

    char wpath[1024], bpath[1024];
    nova_path_join(wpath, sizeof(wpath), dirpath, "weights.nova");
    nova_path_join(bpath, sizeof(bpath), dirpath, "biases.nova");

    FILE* wf = fopen(wpath, "rb");
    FILE* bf = fopen(bpath, "rb");
    if (!wf || !bf) {
        free(wf ? 0 : (void*)0); if (wf) fclose(wf); if (bf) fclose(bf);
        free(cache->layers); free(cache); return NOVA_ERROR_INCOMPLETE_MODEL;
    }

    for (size_t i = 0; i < model->layer_count; i++) {
        NOVALayerParams* P = &cache->layers[i];
        uint32_t wc, bc;

        if (fread(&wc, sizeof(wc), 1, wf) != 1 ||
            fread(&bc, sizeof(bc), 1, bf) != 1) {
            fclose(wf); fclose(bf); return NOVA_ERROR_FORMAT;
        }

        P->weight_count = (size_t)wc;
        P->bias_count = (size_t)bc;
        P->weights = (float*)malloc(P->weight_count * sizeof(float));
        P->biases = (float*)malloc(P->bias_count * sizeof(float));
        P->grad_weights = (float*)calloc(P->weight_count, sizeof(float));
        P->grad_biases = (float*)calloc(P->bias_count, sizeof(float));

        if (!P->weights || !P->biases || !P->grad_weights || !P->grad_biases) {
            fclose(wf); fclose(bf); return NOVA_ERROR_OUT_OF_MEMORY;
        }

        if (fread(P->weights, sizeof(float), P->weight_count, wf) != P->weight_count ||
            fread(P->biases, sizeof(float), P->bias_count, bf) != P->bias_count) {
            fclose(wf); fclose(bf); return NOVA_ERROR_FORMAT;
        }
    }
    fclose(wf);
    fclose(bf);

    model->cache = cache;
    return NOVA_SUCCESS;
}

NOVA_Status nova_verify(const char* dirpath) {
    if (!dirpath) return NOVA_ERROR_INVALID_ARGUMENT;

    char hash_path[1024];
    nova_path_join(hash_path, sizeof(hash_path), dirpath, "checksum.sha256");

    FILE* f = fopen(hash_path, "r");
    if (!f) {
        nova_set_last_error("checksum.sha256 not found");
        return NOVA_ERROR_INCOMPLETE_MODEL;
    }

    char line[1024];
    int errors = 0;
    while (fgets(line, sizeof(line), f)) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (strlen(line) < 65) continue;

        char expected_hex[65];
        char filename[512];
        memcpy(expected_hex, line, 64);
        expected_hex[64] = '\0';
        const char* fn = line + 65;
        while (*fn == ' ') fn++;
        nova_path_join(filename, sizeof(filename), dirpath, fn);

        unsigned char expected[32];
        if (nova_hash_from_hex(expected_hex, expected) != NOVA_SUCCESS) {
            nova_set_last_error("invalid checksum format");
            errors++;
            continue;
        }

        if (nova_verify_file_hash(filename, expected) != NOVA_SUCCESS) {
            char errmsg[256];
            snprintf(errmsg, sizeof(errmsg), "checksum mismatch: %s", fn);
            nova_set_last_error(errmsg);
            errors++;
        }
    }
    fclose(f);

    return errors ? NOVA_ERROR_CHECKSUM_MISMATCH : NOVA_SUCCESS;
}
