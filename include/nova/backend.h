#ifndef NOVA_BACKEND_H
#define NOVA_BACKEND_H

#include <stddef.h>
#include <nova/errors.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NOVA_BACKEND_CPU = 0,
    NOVA_BACKEND_CUDA = 1,
    NOVA_BACKEND_ROCM = 2,
    NOVA_BACKEND_OPENCL = 3,
    NOVA_BACKEND_VULKAN = 4,
    NOVA_BACKEND_ONEAPI = 5,
    NOVA_BACKEND_COUNT
} NOVA_BackendType;

typedef struct NOVABackend NOVABackend;

typedef struct {
    NOVA_BackendType type;
    const char* name;
    int (*available)(void);
    NOVA_Status (*init)(void);
    void (*shutdown)(void);
    NOVA_Status (*gemm)(char trans_a, char trans_b, size_t m, size_t n, size_t k,
                        float alpha, const float* A, size_t lda,
                        const float* B, size_t ldb, float beta,
                        float* C, size_t ldc);
    NOVA_Status (*mem_alloc)(void** ptr, size_t size);
    NOVA_Status (*mem_free)(void* ptr);
    NOVA_Status (*mem_copy)(void* dst, const void* src, size_t size);
    NOVA_Status (*train_forward)(void* model_ctx, const float* input,
                                 float** outputs, size_t* out_sizes);
} NOVABackendVTable;

NOVA_Status nova_backend_init(void);
void nova_backend_shutdown(void);
NOVA_Status nova_backend_select(NOVA_BackendType type);
NOVA_BackendType nova_backend_active(void);
const char* nova_backend_name(NOVA_BackendType type);
int nova_backend_is_available(NOVA_BackendType type);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_BACKEND_H */
