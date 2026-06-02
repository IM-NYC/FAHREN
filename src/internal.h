#ifndef INTERNAL_H
#define INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <fahren/fahren.h>
#include <fahren/utils/optimizers.h>

#define FAHREN_FILE_MAGIC  0x46414852u
#define FAHREN_FILE_VERSION 1u

typedef struct FahrenFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_count;
    uint32_t input_dim;
} FahrenFileHeader;

typedef struct FAHRENLayer {
    int density;
    int layer_type;
    int activation;

    size_t input_size;
    size_t output_size;

    long weights_offset;
    long bias_offset;

    struct FAHRENModel* sub_model;
    int param1;
    int param2;
} FAHRENLayer;

typedef struct FAHRENLayerParams {
    float* weights;
    float* biases;
    float* grad_weights;
    float* grad_biases;
    FAHRENOptimizerState* opt_state_w;
    FAHRENOptimizerState* opt_state_b;
    size_t weight_count;
    size_t bias_count;
} FAHRENLayerParams;

typedef struct FAHRENWeightCache {
    FAHRENLayerParams* layers;
    size_t layer_count;
    char* filepath;
    int loaded;
    int dirty;
} FAHRENWeightCache;

struct FAHRENModel {
    int initialized;
    int finalized;

    size_t layer_count;
    int model_type;
    FAHRENLayer* layers;
    size_t current_layer;

    size_t input_dim;
    char* weights_path;
    FAHRENWeightCache* cache;
};

size_t fahren_random_bytes(void* buf, size_t n);
float  fahren_rand_uniform(float a, float b);

int _fahren_write_model_binary(struct FAHRENModel* cm, const char* filepath, int input_dim);

int  fahren_weights_load(struct FAHRENModel* cm, const char* filepath);
int  fahren_weights_flush(struct FAHRENModel* cm, const char* filepath);
void fahren_weights_free_cache(struct FAHRENModel* cm);

void fahren_gemm(float trans_a, float trans_b, size_t m, size_t n, size_t k,
                 float alpha, const float* A, size_t lda,
                 const float* B, size_t ldb, float beta, float* C, size_t ldc);

void fahren_zero_layer_grads(struct FAHRENModel* cm);
int  fahren_forward_cached(struct FAHRENModel* cm, const float* x,
                           float*** layer_outputs, size_t* layer_out_sizes);
void fahren_free_layer_outputs(float** outs, size_t L);
int  fahren_backward_accumulate(struct FAHRENModel* cm, float** layer_outputs,
                                size_t* layer_out_sizes, int label);
void fahren_apply_layer_gradients(struct FAHRENModel* cm, const FAHRENTrainConfig* config,
                                  size_t batch_size, size_t iteration);
void fahren_ensure_optimizer_states(struct FAHRENModel* cm, const FAHRENOptimizer* opt);

int fahren_train_cpu(struct FAHRENModel* cm, const float* inputs, size_t sample_count,
                     size_t input_dim, const int* labels, size_t num_classes,
                     const char* weights_path, size_t epochs,
                     const FAHRENTrainConfig* config);

#ifdef FAHREN_ENABLE_CUDA
#ifdef __cplusplus
extern "C" {
#endif
int fahren_train_cuda(struct FAHRENModel* cm, const float* inputs, size_t sample_count,
                      size_t input_dim, const int* labels, size_t num_classes,
                      const char* weights_path, size_t epochs,
                      const FAHRENTrainConfig* config);
int fahren_cuda_init(void);
#ifdef __cplusplus
}
#endif
#endif

#endif /* INTERNAL_H */
