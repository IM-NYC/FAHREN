#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <nova/errors.h>
#include "internal.h"

/* SHA-256 implementation (FIPS 180-4) */

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} SHA256_CTX;

static void sha256_transform(SHA256_CTX* ctx, const uint8_t block[64]) {
    uint32_t W[64], a, b, c, d, e, f, g, h, t1, t2;
    for (int i = 0; i < 16; i++)
        W[i] = ((uint32_t)block[4*i]) << 24 | ((uint32_t)block[4*i+1]) << 16 |
               ((uint32_t)block[4*i+2]) << 8 | block[4*i+3];
    for (int i = 16; i < 64; i++)
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e,f,g) + K[i] + W[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(SHA256_CTX* ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
    memset(ctx->buffer, 0, 64);
}

static void sha256_update(SHA256_CTX* ctx, const uint8_t* data, size_t len) {
    size_t idx = (size_t)(ctx->count & 63);
    ctx->count += (uint64_t)len;
    size_t part = 64 - idx;
    if (len >= part) {
        memcpy(ctx->buffer + idx, data, part);
        sha256_transform(ctx, ctx->buffer);
        for (size_t i = part; i + 63 < len; i += 64)
            sha256_transform(ctx, data + i);
        idx = 0;
    } else {
        memcpy(ctx->buffer + idx, data, len);
        return;
    }
    memcpy(ctx->buffer, data + (len - idx), idx);
}

static void sha256_final(SHA256_CTX* ctx, uint8_t hash[32]) {
    uint64_t bits = ctx->count * 8;
    size_t idx = (size_t)(ctx->count & 63);
    size_t pad = (idx < 56) ? (56 - idx) : (120 - idx);
    uint8_t padding[64];
    memset(padding, 0, 64);
    padding[0] = 0x80;
    sha256_update(ctx, padding, pad);
    uint8_t len_bytes[8];
    for (int i = 0; i < 8; i++)
        len_bytes[i] = (uint8_t)(bits >> (56 - 8*i));
    sha256_update(ctx, len_bytes, 8);
    for (int i = 0; i < 8; i++) {
        hash[4*i]   = (uint8_t)(ctx->state[i] >> 24);
        hash[4*i+1] = (uint8_t)(ctx->state[i] >> 16);
        hash[4*i+2] = (uint8_t)(ctx->state[i] >> 8);
        hash[4*i+3] = (uint8_t)(ctx->state[i]);
    }
}

void nova_hash_buffer(const unsigned char* data, size_t len, unsigned char hash[32]) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash);
}

NOVA_Status nova_hash_file(const char* path, unsigned char hash[32]) {
    if (!path || !hash) return NOVA_ERROR_INVALID_ARGUMENT;

    FILE* f = fopen(path, "rb");
    if (!f) {
        nova_set_last_error("cannot open file for hashing");
        return NOVA_ERROR_IO;
    }

    SHA256_CTX ctx;
    sha256_init(&ctx);

    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        sha256_update(&ctx, buf, n);

    if (ferror(f)) {
        fclose(f);
        nova_set_last_error("read error during hashing");
        return NOVA_ERROR_IO;
    }

    sha256_final(&ctx, hash);
    fclose(f);
    return NOVA_SUCCESS;
}

NOVA_Status nova_verify_file_hash(const char* path, const unsigned char expected[32]) {
    unsigned char actual[32];
    NOVA_Status rc = nova_hash_file(path, actual);
    if (rc != NOVA_SUCCESS) return rc;

    for (int i = 0; i < 32; i++) {
        if (actual[i] != expected[i]) {
            nova_set_last_error("file hash does not match expected checksum");
            return NOVA_ERROR_CHECKSUM_MISMATCH;
        }
    }
    return NOVA_SUCCESS;
}

void nova_hash_to_hex(const unsigned char hash[32], char hex[65]) {
    for (int i = 0; i < 32; i++)
        snprintf(hex + i*2, 3, "%02x", hash[i]);
    hex[64] = '\0';
}

NOVA_Status nova_hash_from_hex(const char hex[65], unsigned char hash[32]) {
    if (!hex) return NOVA_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < 32; i++) {
        unsigned int byte;
        if (sscanf(hex + i*2, "%2x", &byte) != 1)
            return NOVA_ERROR_INVALID_ARGUMENT;
        hash[i] = (unsigned char)byte;
    }
    return NOVA_SUCCESS;
}
