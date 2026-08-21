#include <stdlib.h>
#include <string.h>

#include <nova/backend.h>
#include <nova/errors.h>
#include "internal.h"

static NOVA_BackendType g_active_backend = NOVA_BACKEND_CPU;

static int cpu_available(void) { return 1; }
static NOVA_Status cpu_init(void) { return NOVA_SUCCESS; }
static void cpu_shutdown(void) {}

static NOVA_Status cpu_gemm(char trans_a, char trans_b, size_t m, size_t n, size_t k,
                             float alpha, const float* A, size_t lda,
                             const float* B, size_t ldb, float beta,
                             float* C, size_t ldc) {
    nova_gemm(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return NOVA_SUCCESS;
}

static NOVA_Status cpu_mem_alloc(void** ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr ? NOVA_SUCCESS : NOVA_ERROR_OUT_OF_MEMORY;
}

static NOVA_Status cpu_mem_free(void* ptr) {
    free(ptr);
    return NOVA_SUCCESS;
}

static NOVA_Status cpu_mem_copy(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
    return NOVA_SUCCESS;
}

static const NOVABackendVTable cpu_backend = {
    .type = NOVA_BACKEND_CPU,
    .name = "cpu",
    .available = cpu_available,
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .gemm = cpu_gemm,
    .mem_alloc = cpu_mem_alloc,
    .mem_free = cpu_mem_free,
    .mem_copy = cpu_mem_copy,
    .train_forward = NULL
};

static const NOVABackendVTable* g_backends[NOVA_BACKEND_COUNT];

NOVA_Status nova_backend_init(void) {
    g_backends[NOVA_BACKEND_CPU] = &cpu_backend;

    for (int i = 0; i < NOVA_BACKEND_COUNT; i++) {
        if (g_backends[i] && g_backends[i]->available) {
            g_backends[i]->init();
        }
    }

    g_active_backend = NOVA_BACKEND_CPU;
    return NOVA_SUCCESS;
}

void nova_backend_shutdown(void) {
    for (int i = 0; i < NOVA_BACKEND_COUNT; i++) {
        if (g_backends[i] && g_backends[i]->shutdown)
            g_backends[i]->shutdown();
    }
}

NOVA_Status nova_backend_select(NOVA_BackendType type) {
    if (type < 0 || type >= NOVA_BACKEND_COUNT)
        return NOVA_ERROR_INVALID_ARGUMENT;
    if (!g_backends[type] || !g_backends[type]->available())
        return NOVA_ERROR_BACKEND_UNAVAILABLE;
    g_active_backend = type;
    return NOVA_SUCCESS;
}

NOVA_BackendType nova_backend_active(void) {
    return g_active_backend;
}

const char* nova_backend_name(NOVA_BackendType type) {
    switch (type) {
        case NOVA_BACKEND_CPU:    return "cpu";
        case NOVA_BACKEND_CUDA:   return "cuda";
        case NOVA_BACKEND_ROCM:   return "rocm";
        case NOVA_BACKEND_OPENCL: return "opencl";
        case NOVA_BACKEND_VULKAN: return "vulkan";
        case NOVA_BACKEND_ONEAPI: return "oneapi";
        default:                  return "unknown";
    }
}

int nova_backend_is_available(NOVA_BackendType type) {
    if (type < 0 || type >= NOVA_BACKEND_COUNT)
        return 0;
    if (!g_backends[type]) return 0;
    return g_backends[type]->available();
}
