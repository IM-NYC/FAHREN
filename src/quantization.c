#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <nova/quantization.h>
#include <nova/errors.h>

uint16_t nova_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;

    if (exp == 0) {
        /* zero/subnormal */
        return (uint16_t)((sign << 15) | 0);
    }
    if (exp == 0xFF) {
        /* inf/nan */
        return (uint16_t)((sign << 15) | 0x7C00 | (mant >> 13));
    }

    /* Convert to FP16 exponent bias (15 instead of 127) */
    int32_t newexp = (int32_t)exp - 127 + 15;
    if (newexp >= 31) {
        /* overflow to inf */
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    if (newexp <= 0) {
        /* subnormal */
        return (uint16_t)((sign << 15) | 0);
    }

    return (uint16_t)((sign << 15) | (newexp << 10) | (mant >> 13));
}

float nova_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        /* zero/subnormal */
        uint32_t x = (sign << 31) | 0;
        float f;
        memcpy(&f, &x, 4);
        return f;
    }
    if (exp == 0x1F) {
        /* inf/nan */
        uint32_t x = (sign << 31) | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &x, 4);
        return f;
    }

    uint32_t newexp = (uint32_t)((int32_t)exp - 15 + 127);
    uint32_t x = (sign << 31) | (newexp << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

NOVA_Status nova_quantize_fp32_to_int8(const float* src, int8_t* dst,
                                       size_t n, NOVAQuantParams* params) {
    if (!src || !dst || !params || params->num_groups == 0)
        return NOVA_ERROR_INVALID_ARGUMENT;

    size_t group_size = params->group_size;
    if (group_size == 0) group_size = n;

    for (size_t g = 0; g < params->num_groups; g++) {
        size_t start = g * group_size;
        size_t end = start + group_size;
        if (end > n) end = n;
        if (start >= n) break;

        float min = src[start], max = src[start];
        for (size_t i = start + 1; i < end; i++) {
            if (src[i] < min) min = src[i];
            if (src[i] > max) max = src[i];
        }

        /* Symmetric quantization: zero_point = 0, scale = max(|min|,|max|) / 127 */
        float abs_max = fmaxf(fabsf(min), fabsf(max));
        if (abs_max < 1e-10f) abs_max = 1e-10f;

        float scale = abs_max / 127.0f;
        params->scale_factors[g] = scale;
        params->zero_points[g] = 0;

        for (size_t i = start; i < end; i++) {
            float q = roundf(src[i] / scale);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            dst[i] = (int8_t)q;
        }
    }

    return NOVA_SUCCESS;
}

NOVA_Status nova_quantize_int8_to_fp32(const int8_t* src, float* dst,
                                       size_t n, NOVAQuantParams* params) {
    if (!src || !dst || !params || params->num_groups == 0)
        return NOVA_ERROR_INVALID_ARGUMENT;

    size_t group_size = params->group_size;
    if (group_size == 0) group_size = n;

    for (size_t g = 0; g < params->num_groups; g++) {
        size_t start = g * group_size;
        size_t end = start + group_size;
        if (end > n) end = n;
        if (start >= n) break;

        float scale = params->scale_factors[g];
        int32_t zp = params->zero_points[g];

        for (size_t i = start; i < end; i++)
            dst[i] = scale * (float)((int32_t)src[i] - zp);
    }

    return NOVA_SUCCESS;
}

NOVA_Status nova_quantize_fp32_to_fp16(const float* src, uint16_t* dst, size_t n) {
    if (!src || !dst) return NOVA_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < n; i++)
        dst[i] = nova_fp32_to_fp16(src[i]);
    return NOVA_SUCCESS;
}

NOVA_Status nova_quantize_fp16_to_fp32(const uint16_t* src, float* dst, size_t n) {
    if (!src || !dst) return NOVA_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < n; i++)
        dst[i] = nova_fp16_to_fp32(src[i]);
    return NOVA_SUCCESS;
}

NOVA_Status nova_quant_calc_info(const float* data, size_t n, NOVAQuantInfo* info) {
    if (!data || !info) return NOVA_ERROR_INVALID_ARGUMENT;
    if (n == 0) return NOVA_ERROR_INVALID_ARGUMENT;

    float min = data[0], max = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }

    float abs_max = fmaxf(fabsf(min), fabsf(max));
    info->dtype = NOVA_DTYPE_INT8;
    info->scale = (abs_max < 1e-10f) ? 1.0f : abs_max / 127.0f;
    info->zero_point = 0;
    info->num_elements = n;
    return NOVA_SUCCESS;
}

NOVA_Status nova_quant_params_create(NOVAQuantParams* params, size_t num_groups,
                                     size_t group_size) {
    if (!params || num_groups == 0) return NOVA_ERROR_INVALID_ARGUMENT;

    params->scale_factors = (float*)calloc(num_groups, sizeof(float));
    params->zero_points = (int32_t*)calloc(num_groups, sizeof(int32_t));
    if (!params->scale_factors || !params->zero_points) {
        free(params->scale_factors);
        free(params->zero_points);
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    params->num_groups = num_groups;
    params->group_size = group_size;
    params->dtype = NOVA_DTYPE_INT8;
    return NOVA_SUCCESS;
}

void nova_quant_params_destroy(NOVAQuantParams* params) {
    if (!params) return;
    free(params->scale_factors);
    free(params->zero_points);
    memset(params, 0, sizeof(*params));
}
