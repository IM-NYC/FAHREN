#include <stdlib.h>
#include <string.h>

#include <fahren/fahren.h>
#include "internal.h"

static int g_fahren_device = FAHREN_DEVICE_CPU;

int fahren_set_device(int device) {
    if (device != FAHREN_DEVICE_CPU && device != FAHREN_DEVICE_CUDA) {
        return FAHREN_ERROR_INVALID_ARGUMENT;
    }
#ifdef FAHREN_ENABLE_CUDA
    if (device == FAHREN_DEVICE_CUDA && !fahren_cuda_available()) {
        return FAHREN_ERROR_UNSUPPORTED;
    }
#else
    if (device == FAHREN_DEVICE_CUDA) {
        return FAHREN_ERROR_UNSUPPORTED;
    }
#endif
    g_fahren_device = device;
    return FAHREN_SUCCESS;
}

int fahren_get_device(void) {
    const char* env = getenv("FAHREN_DEVICE");
    if (env) {
        if (strcmp(env, "cuda") == 0 || strcmp(env, "gpu") == 0) {
            return FAHREN_DEVICE_CUDA;
        }
    }
    return g_fahren_device;
}

int fahren_cuda_available(void) {
#ifdef FAHREN_ENABLE_CUDA
    return fahren_cuda_init();
#else
    return 0;
#endif
}

int fahren_train_resolve_device(const FAHRENTrainConfig* config) {
    if (config && config->device >= 0) {
        return config->device;
    }
    return fahren_get_device();
}
