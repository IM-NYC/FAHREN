#include <stddef.h>

#include "internal.h"

#ifdef FAHREN_USE_OPENBLAS
#include <cblas.h>
#endif

void fahren_gemm(float trans_a, float trans_b, size_t m, size_t n, size_t k,
                 float alpha, const float* A, size_t lda,
                 const float* B, size_t ldb, float beta, float* C, size_t ldc) {
#ifdef FAHREN_USE_OPENBLAS
    const CBLAS_TRANSPOSE ta = (trans_a != 0) ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = (trans_b != 0) ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb, (int)m, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
    return;
#else
    (void)lda;
    (void)ldb;
    (void)ldc;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                float a_val = (trans_a != 0) ? A[p * m + i] : A[i * k + p];
                float b_val = (trans_b != 0) ? B[j * k + p] : B[p * n + j];
                sum += a_val * b_val;
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
#endif
}
