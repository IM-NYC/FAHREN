#ifndef INTERNAL_H
#define INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <fahren/fahren.h>

/* Internal representation of a layer in a sequential model. */
typedef struct FAHRENLayer {
    int density;                 /* number of neurons / filters (outputs) */
    int layer_type;              /* FAHREN_LAYER_DENSE, etc. */
    int activation;              /* FAHREN_LAYER_ACTIVATION_* */

    size_t input_size;           /* number of inputs to this layer */
    size_t output_size;          /* same as density for dense layers */

    long weights_offset;         /* file offset for weights (float[output][input]) */
    long bias_offset;            /* file offset for biases (float[output]) */

    struct FAHRENModel* sub_model;  /* for nested models, if any (unused in training) */
    int param1;                  /* kernel_size for CONV, pool_size for POOLING */
    int param2;                  /* stride for CONV/POOLING */
} FAHRENLayer;

/* Concrete model structure (completes the opaque type from the public header). */
struct FAHRENModel {
    int initialized;
    int finalized;               /* binary weights file created */

    size_t layer_count;
    int model_type;
    FAHRENLayer* layers;
    size_t current_layer;        /* next layer index to fill */

    size_t input_dim;            /* input dimensionality used when finalizing */
    char* weights_path;          /* path to the binary weights file */
};

/* Random helpers provided by POSIX backend. */
size_t fahren_random_bytes(void* buf, size_t n);
float  fahren_rand_uniform(float a, float b);

/* Internal I/O helpers (init.c) */
int _fahren_write_model_binary(struct FAHRENModel* cm, const char* filepath, int input_dim);

#endif /* INTERNAL_H */
