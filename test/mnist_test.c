#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <nova/nova.h>
#include <nova/mnist.h>
#include <nova/errors.h>
#include "test_harness.h"

extern NOVA_Status nova_path_join(char* buf, size_t sz, const char* dir, const char* file);
extern void nova_path_native(char* path);

#ifndef NOVA_MNIST_DIR
#define NOVA_MNIST_DIR "../mnist"
#endif

#define WEIGHTS "nova_mnist.bin"

static size_t env_size_t(const char* name, size_t def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    return (size_t)strtoul(v, NULL, 10);
}

static void display_dir(const char* dir) {
    if (!dir) { printf("(null)"); return; }
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", dir);
    nova_path_native(buf);
    printf("%s", buf);
}

int main(void) {
    const char* dir = getenv("NOVA_MNIST_DIR");
    if (!dir || !*dir) dir = NOVA_MNIST_DIR;

    float* train_x = NULL;
    int* train_y = NULL;
    size_t train_n = 0;
    float* test_x = NULL;
    int* test_y = NULL;
    size_t test_n = 0;

    NOVA_Status rc = nova_mnist_load_train(dir, &train_x, &train_y, &train_n);
    if (rc != NOVA_SUCCESS) {
        printf("SKIP: MNIST training set not found in ");
        display_dir(dir);
        printf("\n");
        return 0;
    }

    rc = nova_mnist_load_test(dir, &test_x, &test_y, &test_n);
    if (rc != NOVA_SUCCESS) {
        printf("SKIP: MNIST test set not found in ");
        display_dir(dir);
        printf("\n");
        nova_mnist_free(train_x, train_y);
        return 0;
    }

    NOVA_ASSERT(train_n > 0 && test_n > 0, "mnist load");

    size_t cap = env_size_t("NOVA_MNIST_TRAIN_SAMPLES", 10000);
    if (cap > train_n) cap = train_n;
    size_t epochs = env_size_t("NOVA_MNIST_EPOCHS", 10);

    remove(WEIGHTS);

    NOVAModel* m = nova_model_create(NOVA_MODEL_SEQUENTIAL, 3);
    NOVA_ASSERT(m != NULL, "create model");
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 128);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 64);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_SOFTMAX, 10);
    NOVA_ASSERT(nova_model_finalize(m, WEIGHTS, 784) == NOVA_SUCCESS, "finalize");

    NOVATrainConfig cfg = nova_train_config_default(0.01f);
    cfg.batch_size = 64;
    NOVA_ASSERT(nova_train_with_config(m, train_x, cap, 784, train_y, 10, WEIGHTS, epochs, &cfg)
                == NOVA_SUCCESS, "train");

    float acc = 0.0f;
    NOVA_ASSERT(nova_evaluate(m, WEIGHTS, test_x, test_y, test_n, 784, &acc) == NOVA_SUCCESS, "evaluate");
    printf("accuracy: %.2f%% (%zu samples)\n", acc * 100.0f, test_n);

    nova_mnist_free(train_x, train_y);
    nova_mnist_free(test_x, test_y);
    nova_model_destroy(m);
    remove(WEIGHTS);

    NOVA_ASSERT(acc >= 0.10f, "accuracy > 10% (random baseline)");
    printf("\n%d passed, %d failed\n",
           g_nova_tests_run - g_nova_tests_failed, g_nova_tests_failed);
    return g_nova_tests_failed > 0 ? 1 : 0;
}
