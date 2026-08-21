#ifndef NOVA_QUANTIZATION_H
#define NOVA_QUANTIZATION_H

#include <stddef.h>
#include <stdint.h>
#include <nova/errors.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NOVA_DTYPE_FP32 = 0,
    NOVA_DTYPE_FP16 = 1,
    NOVA_DTYPE_INT8 = 2,
    NOVA_DTYPE_INT4 = 3
} NOVA_DType;

typedef struct {
    NOVA_DType dtype;
    float scale;
    int32_t zero_point;
    size_t num_elements;
} NOVAQuantInfo;

typedef struct {
    NOVA_DType dtype;
    float* scale_factors;
    int32_t* zero_points;
    size_t num_groups;
    size_t group_size;
} NOVAQuantParams;

NOVA_Status nova_quantize_fp32_to_int8(const float* src, int8_t* dst,
                                       size_t n, NOVAQuantParams* params);
NOVA_Status nova_quantize_int8_to_fp32(const int8_t* src, float* dst,
                                       size_t n, NOVAQuantParams* params);
NOVA_Status nova_quantize_fp32_to_fp16(const float* src, uint16_t* dst, size_t n);
NOVA_Status nova_quantize_fp16_to_fp32(const uint16_t* src, float* dst, size_t n);

NOVA_Status nova_quant_calc_info(const float* data, size_t n, NOVAQuantInfo* info);
NOVA_Status nova_quant_params_create(NOVAQuantParams* params, size_t num_groups,
                                     size_t group_size);
void nova_quant_params_destroy(NOVAQuantParams* params);

uint16_t nova_fp32_to_fp16(float f);
float nova_fp16_to_fp32(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_QUANTIZATION_H */
