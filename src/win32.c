/*
 * Windows platform backend for FAHREN.
 * Cryptographically suitable random bytes when available; rand() fallback.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "internal.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

size_t fahren_random_bytes(void* buf, size_t n) {
    if (!buf || n == 0) {
        return 0;
    }

#if defined(_WIN32)
    NTSTATUS status = BCryptGenRandom(
        NULL,
        (PUCHAR)buf,
        (ULONG)n,
        BCRYPT_USE_SYSTEM_PREFERRED_RNG
    );
    if (status == 0) {
        return n;
    }
#endif

    uint8_t* p = (uint8_t*)buf;
    for (size_t i = 0; i < n; ++i) {
        p[i] = (uint8_t)(rand() & 0xFF);
    }
    return n;
}

static inline float u32_to_unit(uint32_t x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

float fahren_rand_uniform(float a, float b) {
    uint32_t x = 0;
    if (fahren_random_bytes(&x, sizeof(x)) != sizeof(x)) {
        x = (uint32_t)rand();
    }
    float u = u32_to_unit(x);
    return a + (b - a) * u;
}
