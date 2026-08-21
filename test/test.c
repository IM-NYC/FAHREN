#include <string.h>

#include <nova/nova.h>
#include <nova/errors.h>
#include <nova/quantization.h>
#include "../src/internal.h"
#include "test_harness.h"

#define WEIGHTS "nova_smoke.bin"

static void test_errors(void) {
    NOVA_ASSERT(strcmp(nova_strerror(NOVA_SUCCESS), "success") == 0, "strerror success");
    nova_set_last_error("detail");
    NOVA_ASSERT(strcmp(nova_last_error_message(), "detail") == 0, "last error");
    nova_clear_last_error();
}

static void test_tiny_train(void) {
    float x[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    int y[] = {0, 1, 0, 1};
    remove(WEIGHTS);

    NOVAModel* m = nova_model_create(NOVA_MODEL_SEQUENTIAL, 2);
    NOVA_ASSERT(m != NULL, "create model");
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 4);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_SOFTMAX, 2);
    NOVA_ASSERT(nova_model_finalize(m, WEIGHTS, 4) == NOVA_SUCCESS, "finalize");

    NOVATrainConfig cfg = nova_train_config_default(0.05f);
    cfg.batch_size = 2;
    NOVA_ASSERT(nova_train_with_config(m, x, 4, 4, y, 2, WEIGHTS, 2, &cfg) == NOVA_SUCCESS, "train");

    nova_model_destroy(m);
    remove(WEIGHTS);
}

static void test_device(void) {
    NOVA_ASSERT(nova_device_get() == NOVA_DEVICE_CPU, "default device cpu");
    NOVA_ASSERT(nova_device_set(NOVA_DEVICE_CPU) == NOVA_SUCCESS, "set cpu");
    NOVA_ASSERT(nova_device_available(NOVA_DEVICE_CPU) == 1, "cpu available");
    NOVA_ASSERT(strcmp(nova_device_name(NOVA_DEVICE_CPU), "cpu") == 0, "device name cpu");
}

static void test_hash(void) {
    unsigned char hash[32];
    char test_path[1024];
    nova_path_join(test_path, sizeof(test_path), NOVA_TEST_DIR, "test.c");
    NOVA_ASSERT(nova_hash_file(test_path, hash) == NOVA_SUCCESS, "hash file");

    char hex[65];
    nova_hash_to_hex(hash, hex);
    NOVA_ASSERT(strlen(hex) == 64, "hash hex length");

    unsigned char hash2[32];
    NOVA_ASSERT(nova_hash_from_hex(hex, hash2) == NOVA_SUCCESS, "hash from hex");
    NOVA_ASSERT(memcmp(hash, hash2, 32) == 0, "hash roundtrip");
}

static void test_quantization(void) {
    extern uint16_t nova_fp32_to_fp16(float f);
    extern float nova_fp16_to_fp32(uint16_t h);

    float orig = 3.14159f;
    uint16_t h = nova_fp32_to_fp16(orig);
    float back = nova_fp16_to_fp32(h);
    NOVA_ASSERT(back > 3.14f && back < 3.15f, "fp16 roundtrip");

    float vals[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int8_t q[5];
    float dq[5];
    NOVAQuantParams params;
    NOVA_ASSERT(nova_quant_params_create(&params, 1, 5) == NOVA_SUCCESS, "quant params");
    NOVA_ASSERT(nova_quantize_fp32_to_int8(vals, q, 5, &params) == NOVA_SUCCESS, "quantize");
    NOVA_ASSERT(nova_quantize_int8_to_fp32(q, dq, 5, &params) == NOVA_SUCCESS, "dequantize");
    for (int i = 0; i < 5; i++)
        NOVA_ASSERT(fabsf(dq[i] - vals[i]) < 0.02f, "quant accuracy");
    nova_quant_params_destroy(&params);
}

int main(void) {
    printf("Novaflow smoke (v%d.%d.%d)\n\n",
           NOVA_VERSION_MAJOR, NOVA_VERSION_MINOR, NOVA_VERSION_PATCH);
    test_errors();
    test_tiny_train();
    test_device();
    test_hash();
    test_quantization();
    printf("\n%d passed, %d failed\n",
           g_nova_tests_run - g_nova_tests_failed, g_nova_tests_failed);
    return g_nova_tests_failed > 0 ? 1 : 0;
}
