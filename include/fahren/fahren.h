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

/* Version */
#define FAHREN_VERSION_MAJOR 1
#define FAHREN_VERSION_MINOR 0
#define FAHREN_VERSION_PATCH 0

/* Simple status codes */
typedef enum FAHRENStatus {
    FAHREN_SUCCESS = 0,
    FAHREN_ERROR_INVALID_ARGUMENT = 1,
    FAHREN_ERROR_NOT_INITIALIZED = 2,
    FAHREN_ERROR_PROCESSING_FAILED = 3
} FAHRENStatus;

/* A minimal model type enum: we only need a placeholder for now. */
typedef enum FAHRENModelType {
    FAHREN_MODEL_SEQUENTIAL = 0
} FAHRENModelType;

/* Layer kinds supported by the tiny API. */
typedef enum FAHRENLayerType {
    FAHREN_LAYER_DENSE = 0,
    FAHREN_LAYER_CONVOLUTIONAL = 1
} FAHRENLayerType;

/* A very small layer descriptor. The user only needs to set `density` and
 * `previous_layer` when building simple sequential models in examples. */
typedef struct FAHRENLayer {
    int density;               /* number of neurons / filters */
    struct FAHRENLayer* previous_layer; /* pointer to previous layer or NULL */
    FAHRENLayerType layer_type;/* kind of layer */
} FAHRENLayer;

/* Opaque model instance held by library users; keep fields minimal. */
typedef struct FAHREN {
    int initialized;
    size_t layer_count;
    FAHRENModelType model_type;
    FAHRENLayer* layers;
} FAHREN;

/* Public API: simple and self-explanatory names. Signatures are intentionally
 * small so users can easily call them from examples. */

/* Allocate `count` zero-initialized FAHRENLayer entries. Free with `free()`.
 * This avoids callers needing to cast `calloc` results in C. */
FAHRENLayer* fahren_alloc_layers(size_t count);

/* Initialize a model instance. Pass a pointer to a FAHREN struct (it will be
 * populated) and a preallocated array of `layer_count` layers. */
FAHRENStatus fahren_init(FAHREN* cm, FAHRENModelType model_type, size_t layer_count, FAHRENLayer* layers);

/* Shutdown and free resources associated with a model. */
FAHRENStatus fahren_shutdown(FAHREN* cm);

/* Note: text-processing and small training/prediction helpers were removed
 * to keep the public API minimal and easy to understand. Implementations
 * are intentionally small and focused; model serialization (`fahren_write_random_weights`)
 * is still supported via the implementation.
 */

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_H */