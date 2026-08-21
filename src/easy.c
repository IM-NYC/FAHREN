#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <nova/nova.h>
#include "internal.h"

NOVAModel* nova_easy_model_dense(int input_dim, const char* path, ...) {
    if (!path || input_dim <= 0) return NULL;

    va_list ap;
    va_start(ap, path);

    int acts[32], units[32];
    size_t layer_count = 0;

    for (;;) {
        int act = va_arg(ap, int);
        if (act == -1) break;
        if (layer_count >= 32) { va_end(ap); return NULL; }
        acts[layer_count] = act;
        units[layer_count] = va_arg(ap, int);
        ++layer_count;
    }
    va_end(ap);

    if (layer_count == 0) return NULL;

    NOVAModel* model = nova_model_create(NOVA_MODEL_SEQUENTIAL, (int)layer_count);
    if (!model) return NULL;

    for (size_t i = 0; i < layer_count; ++i)
        nova_model_add_layer(model, NOVA_LAYER_DENSE, acts[i], units[i]);

    if (nova_model_finalize(model, path, input_dim) != NOVA_SUCCESS) {
        nova_model_destroy(model);
        return NULL;
    }
    return model;
}
