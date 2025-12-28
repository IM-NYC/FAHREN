/* Simple POSIX implementation file for FAHREN.
 * Keep internals local; this file implements the minimal API declared in
 * include/FAHREN/fahren.h. The code below is written to be straightforward
 * and easy for readers to follow. */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <math.h>

#include <fahren/fahren.h>

/* Define the opaque FAHRENModel struct */
struct FAHRENModel {
    int initialized;
    size_t layer_count;
    int model_type;
    FAHRENLayer* layers;
    size_t current_layer;  /* Tracks the next layer position to add */
};

void fahren_add_layer(FAHRENModel* cm, int layer_type, ...) {
    if (!cm || cm->current_layer >= cm->layer_count) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to add layer due to invalid layer count\n");
        #endif
        abort();
    }
    size_t layerpos = cm->current_layer;
    va_list args;
    va_start(args, layer_type);
    FAHRENModel* sub_model = NULL;
    int density = 0;
    int param1 = 0;
    int param2 = 0;
    
    if (layer_type == FAHREN_LAYER_SUBMODEL) {
        sub_model = va_arg(args, FAHRENModel*);
        if (!sub_model || sub_model->layer_count == 0) {
            fahren_throw(FAHREN_ERROR_INVALID_ARGUMENT);
        }
        density = sub_model->layers[sub_model->layer_count - 1].density;
    } else if (layer_type == FAHREN_LAYER_CONVOLUTIONAL) {
        density = va_arg(args, int);  // filters
        param1 = va_arg(args, int);   // kernel_size
        param2 = va_arg(args, int);   // stride
        sub_model = va_arg(args, FAHRENModel*);  // usually NULL
        if (density <= 0 || param1 <= 0 || param2 <= 0) {
            fahren_throw(FAHREN_ERROR_INVALID_ARGUMENT);
        }
    } else if (layer_type == FAHREN_LAYER_POOLING) {
        param1 = va_arg(args, int);   // pool_size
        param2 = va_arg(args, int);   // stride
        sub_model = va_arg(args, FAHRENModel*);  // usually NULL
        if (param1 <= 0 || param2 <= 0) {
            fahren_throw(FAHREN_ERROR_INVALID_ARGUMENT);
        }
    } else {  // DENSE or default
        density = va_arg(args, int);
        sub_model = va_arg(args, FAHRENModel*);  // usually NULL
        if (density <= 0) {
            fahren_throw(FAHREN_ERROR_INVALID_ARGUMENT);
        }
    }
    
    cm->layers[layerpos].density = density;
    cm->layers[layerpos].layer_type = layer_type;
    cm->layers[layerpos].previous_layer = (layerpos > 0) ? &cm->layers[layerpos - 1] : NULL;
    cm->layers[layerpos].sub_model = sub_model;
    cm->layers[layerpos].param1 = param1;
    cm->layers[layerpos].param2 = param2;
    va_end(args);
    cm->current_layer++;
}

FAHRENModel* fahren_create_model(int model_type, int layer_count) {
    if (layer_count == 0) {
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to create model due to invalid layer count\n");
        #endif
        abort();
    }
    FAHRENModel* cm = (FAHRENModel*)malloc(sizeof(FAHRENModel));
    if (!cm) {
        free(cm);
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to create model due to memory allocation failure\n");
        #endif
        abort();
    }
    cm->model_type = model_type;
    cm->layer_count = layer_count;
    cm->layers = (FAHRENLayer*)calloc(layer_count, sizeof(FAHRENLayer));
    if (!cm->layers) {
        free(cm);
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to create model due to memory allocation failure\n");
        #endif
        abort();
    }
    cm->initialized = 1;
    cm->current_layer = 0;

#if FAHREN_VERBOSE
    fprintf(stdout, "FAHREN LOG: Successfully created model with %d initialized layers\n", layer_count);
#endif

    /* TODO: write initial random weights & biases for inspection */
    // _fahren_write_random_weights(cm, "fahren_initial_model.bin");

    return cm;
}

void fahren_shutdown(FAHRENModel* cm) {
    if (!cm){
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to shutdown model due to invalid model pointer\n");
        #endif
        abort();
    } 
    if (!cm->initialized){
        #if FAHREN_VERBOSE
        fprintf(stderr, "FAHREN ERROR: Failed to create model due to invalid layer count\n");
        #endif
        abort();
    }

    /* Free allocated layer array if present */
    if (cm->layers) {
        free(cm->layers);
        cm->layers = NULL;
    }

    /* Free the model itself */
    free(cm);

    return;
}