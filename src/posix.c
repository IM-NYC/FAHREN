#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "internal.h"

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#ifdef __APPLE__
#include <CommonCrypto/CommonRandom.h>
#endif
#ifdef __linux__
#include <sys/random.h>
#endif
#endif

size_t nova_random_bytes(void* buf, size_t n) {
    if (!buf || n == 0) return 0;

#if defined(__APPLE__)
    if (CCRandomGenerateBytes(buf, n) == kCCSuccess) return n;
    arc4random_buf(buf, n);
    return n;
#elif defined(__linux__)
    ssize_t r = getrandom(buf, n, 0);
    if (r == (ssize_t)n) return (size_t)r;
#endif

#if defined(__unix__) || defined(__APPLE__)
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t total = 0;
        while (total < n) {
            ssize_t m = read(fd, (uint8_t*)buf + total, n - total);
            if (m <= 0) { if (errno == EINTR) continue; break; }
            total += (size_t)m;
        }
        close(fd);
        if (total == n) return total;
    }
#endif

    uint8_t* p = (uint8_t*)buf;
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)(rand() & 0xFF);
    return n;
}

static inline float u32_to_unit(uint32_t x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

float nova_rand_uniform(float a, float b) {
    uint32_t x = 0;
    if (nova_random_bytes(&x, sizeof(x)) != sizeof(x)) x = (uint32_t)rand();
    float u = u32_to_unit(x);
    return a + (b - a) * u;
}
