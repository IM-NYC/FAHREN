#include <stddef.h>
#include "internal.h"

#ifdef NOVA_USE_OPENBLAS
#include <cblas.h>
#endif

void nova_gemm(char trans_a, char trans_b, size_t m, size_t n, size_t k,
               float alpha, const float* A, size_t lda,
               const float* B, size_t ldb, float beta, float* C, size_t ldc) {
#ifdef NOVA_USE_OPENBLAS
    CBLAS_TRANSPOSE ta = (trans_a == 'T') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (trans_b == 'T') ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb, (int)m, (int)n, (int)k,
                alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
#else
    (void)lda; (void)ldb; (void)ldc;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                float a_val = (trans_a == 'T') ? A[p * m + i] : A[i * k + p];
                float b_val = (trans_b == 'T') ? B[j * k + p] : B[p * n + j];
                sum += a_val * b_val;
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
#endif
}
