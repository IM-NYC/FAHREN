#ifndef NOVA_INTERNAL_H
#define NOVA_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include <nova/nova.h>
#include <nova/types.h>

#define NOVA_FILE_MAGIC  0x4E4F5641u
#define NOVA_FILE_VERSION 1u

typedef struct NovaFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_count;
    uint32_t input_dim;
} NovaFileHeader;

typedef struct NOVAWeightCache {
    NOVALayerParams* layers;
    size_t layer_count;
    char* dirpath;
    int loaded;
    int dirty;
} NOVAWeightCache;

struct NOVAModel {
    int initialized;
    int finalized;
    size_t layer_count;
    int model_type;
    NOVALayer* layers;
    size_t current_layer;
    size_t input_dim;
    char* path;
    NOVAWeightCache* cache;
};

size_t nova_random_bytes(void* buf, size_t n);
float  nova_rand_uniform(float a, float b);

NOVA_Status nova_weights_load(NOVAModel* model, const char* path);
NOVA_Status nova_weights_flush(NOVAModel* model, const char* path);
void nova_weights_free_cache(NOVAModel* model);

void nova_gemm(char trans_a, char trans_b, size_t m, size_t n, size_t k,
               float alpha, const float* A, size_t lda,
               const float* B, size_t ldb, float beta, float* C, size_t ldc);

void nova_zero_grads(NOVAModel* model);
int  nova_forward(NOVAModel* model, const float* x,
                  float*** layer_outputs, size_t* layer_out_sizes);
void nova_free_outputs(float** outs, size_t L);
int  nova_backward(NOVAModel* model, float** layer_outputs,
                   size_t* layer_out_sizes, int label);
void nova_apply_grads(NOVAModel* model, const NOVATrainConfig* config,
                      size_t batch_size, size_t iteration);
void nova_ensure_opt_states(NOVAModel* model, const NOVAOptimizer* opt);

NOVA_Status nova_train_cpu(NOVAModel* model, const float* inputs, size_t sample_count,
                           size_t input_dim, const int* labels, size_t num_classes,
                           const char* path, size_t epochs,
                           const NOVATrainConfig* config);

NOVA_Status nova_write_binary(NOVAModel* model, const char* path, int input_dim);
int nova_read_header(FILE* f, NovaFileHeader* out);

typedef struct {
    uint32_t input_dim;
    uint32_t layer_count;
    uint64_t layer_types_offset;
    uint64_t weights_offset;
    uint64_t biases_offset;
    uint64_t metadata_offset;
} NovaModelIndex;

NOVA_Status nova_save_multi_file(NOVAModel* model, const char* dirpath);
NOVA_Status nova_load_multi_file(NOVAModel* model, const char* dirpath);

NOVA_Status nova_hash_file(const char* path, unsigned char hash[32]);
NOVA_Status nova_verify_file_hash(const char* path, const unsigned char expected[32]);
void nova_hash_buffer(const unsigned char* data, size_t len, unsigned char hash[32]);
void nova_hash_to_hex(const unsigned char hash[32], char hex[65]);
NOVA_Status nova_hash_from_hex(const char hex[65], unsigned char hash[32]);

NOVA_Status nova_quantize_weights(NOVAModel* model, int precision);
NOVA_Status nova_dequantize_weights(NOVAModel* model);

/* Platform path separator */
#if defined(_WIN32)
#  define NOVA_PATH_SEP_CHAR '\\'
#  define NOVA_PATH_SEP_STR "\\"
#else
#  define NOVA_PATH_SEP_CHAR '/'
#  define NOVA_PATH_SEP_STR "/"
#endif

NOVA_Status nova_path_join(char* buf, size_t sz, const char* dir, const char* file);
void nova_path_native(char* path);

#ifdef NOVA_ENABLE_CUDA
int nova_cuda_init(void);
NOVA_Status nova_train_cuda(NOVAModel* model, const float* inputs, size_t sample_count,
                            size_t input_dim, const int* labels, size_t num_classes,
                            const char* path, size_t epochs,
                            const NOVATrainConfig* config);
#endif

#endif /* NOVA_INTERNAL_H */
