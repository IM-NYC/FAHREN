#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#ifdef __APPLE__
#include <CommonCrypto/CommonRandom.h>
#endif
#ifdef __linux__
#include <sys/random.h>
#endif
#include <math.h>

#include "internal.h"

size_t fahren_random_bytes(void* buf, size_t n) {
    if (!buf || n == 0) return 0;

#ifdef __APPLE__
    if (CCRandomGenerateBytes(buf, n) == kCCSuccess) {
        return n;
    }
#endif
#ifdef __linux__
    ssize_t r = getrandom(buf, n, 0);
    if (r == (ssize_t)n) return (size_t)r;
#endif

#ifdef __APPLE__
    /* arc4random_buf is available on Apple platforms */
    arc4random_buf(buf, n);
    return n;
#endif

    /* Fallback to /dev/urandom */
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t total = 0;
        while (total < n) {
            ssize_t m = read(fd, (uint8_t*)buf + total, n - total);
            if (m <= 0) {
                if (errno == EINTR) continue;
                break;
            }
            total += (size_t)m;
        }
        close(fd);
        if (total == n) return total;
    }

    /* Last resort: stdlib rand() scaled (not cryptographically strong) */
    uint8_t* p = (uint8_t*)buf;
    for (size_t i = 0; i < n; ++i) {
        p[i] = (uint8_t)(rand() & 0xFF);
    }
    return n;
}

static inline float u32_to_unit(uint32_t x) {
    /* Convert to [0,1) using 24 bits of precision */
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
