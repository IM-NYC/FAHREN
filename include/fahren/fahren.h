/*
 * MIT License
 *
 * Copyright (c) 2025 Imran Mukhiddinov <imranmukhiddinov2009@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Minimal, easy-to-read public header for the FAHREN library.
 * This header exposes only the small API users need to build and run
 * simple examples: create layers, initialize a model, do text processing,
 * train a tiny softmax classifier, predict from a saved model, and
 * clean up. Keep this file short so it's quick to scan.
 */
#ifndef FAHREN_H
#define FAHREN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h> /* for fprintf in FAHREN_THROW */
#include <stdlib.h> /* for abort in FAHREN_THROW */

/* Verbosity control: redefine to 0 to disable verbose output */
#ifndef FAHREN_VERBOSE
#define FAHREN_VERBOSE 1
#endif

/* Version */
#define FAHREN_VERSION_MAJOR                1
#define FAHREN_VERSION_MINOR                0
#define FAHREN_VERSION_PATCH                0
/* Status codes */
#define FAHREN_SUCCESS                      0
#define FAHREN_ERROR_INVALID_ARGUMENT       1
#define FAHREN_ERROR_NOT_INITIALIZED        2
#define FAHREN_ERROR_PROCESSING_FAILED      3

#define FAHREN_MODEL_SEQUENTIAL             0
#define FAHREN_MODEL_LSTM                   1

#define FAHREN_LAYER_DENSE                  0
#define FAHREN_LAYER_CONVOLUTIONAL          1
#define FAHREN_LAYER_POOLING                2
#define FAHREN_LAYER_SUBMODEL               3

#define FAHREN_LAYER_ACTIVATION_RELU        0
#define FAHREN_LAYER_ACTIVATION_SIGMOID     1
#define FAHREN_LAYER_ACTIVATION_TANH        2
#define FAHREN_LAYER_ACTIVATION_SOFTMAX     3

/* Opaque model instance held by library users; keep fields minimal. */
typedef struct FAHRENModel FAHRENModel;

/* A very small layer descriptor. The user only needs to set `density` and
 * `previous_layer` when building simple sequential models in examples. */
typedef struct FAHRENLayer {
    int density;               /* number of neurons / filters */
    int layer_type;           /* FAHREN_LAYER_DENSE, etc. */
    struct FAHRENLayer* previous_layer; /* pointer to previous layer or NULL */
    FAHRENModel* sub_model;      /* for nested models, if any */
    int param1;                /* kernel_size for CONV, pool_size for POOLING */
    int param2;                /* stride for CONV/POOLING */
} FAHRENLayer;

/* Public API: simple and self-explanatory names. Signatures are intentionally
 * small so users can easily call them from examples. */

/* Initialize a model instance. Pass a model type and layer count, returns a pointer to the model. */
FAHRENModel* fahren_create_model(int model_type, int layer_count);

/* Shutdown and free resources associated with a model. */
void fahren_shutdown(FAHRENModel* cm);

/* Add a layer to the model. */
void fahren_add_layer(FAHRENModel* cm, int layer_type, ...);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_H */