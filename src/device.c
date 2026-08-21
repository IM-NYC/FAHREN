#include <stdlib.h>
#include <string.h>

#include <nova/nova.h>
#include "internal.h"

static int g_nova_device = NOVA_DEVICE_CPU;

NOVA_Status nova_device_set(int device) {
    if (device < 0 || device >= NOVA_DEVICE_ONEAPI + 1)
        return NOVA_ERROR_INVALID_ARGUMENT;
    g_nova_device = device;
    return NOVA_SUCCESS;
}

int nova_device_get(void) {
    const char* env = getenv("NOVA_DEVICE");
    if (env) {
        if (strcmp(env, "cuda") == 0) return NOVA_DEVICE_CUDA;
        if (strcmp(env, "rocm") == 0) return NOVA_DEVICE_ROCM;
        if (strcmp(env, "opencl") == 0) return NOVA_DEVICE_OPENCL;
        if (strcmp(env, "vulkan") == 0) return NOVA_DEVICE_VULKAN;
        if (strcmp(env, "oneapi") == 0) return NOVA_DEVICE_ONEAPI;
    }
    return g_nova_device;
}

int nova_device_available(int device) {
    switch (device) {
        case NOVA_DEVICE_CPU:
            return 1;
#ifdef NOVA_ENABLE_CUDA
        case NOVA_DEVICE_CUDA:
            return 1;
#endif
        default:
            return 0;
    }
}

const char* nova_device_name(int device) {
    switch (device) {
        case NOVA_DEVICE_CPU:    return "cpu";
        case NOVA_DEVICE_CUDA:   return "cuda";
        case NOVA_DEVICE_ROCM:   return "rocm";
        case NOVA_DEVICE_OPENCL: return "opencl";
        case NOVA_DEVICE_VULKAN: return "vulkan";
        case NOVA_DEVICE_ONEAPI: return "oneapi";
        default:                 return "unknown";
    }
}
