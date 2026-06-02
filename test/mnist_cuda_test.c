#include <stdlib.h>
#include <stdio.h>

#include <fahren/fahren_easy.h>
#include "test_harness.h"

#ifndef FAHREN_MNIST_DIR
#define FAHREN_MNIST_DIR "../mnist"
#endif

#define WEIGHTS "fahren_mnist_cuda.bin"

static size_t env_size_t(const char* name, size_t def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    return (size_t)strtoul(v, NULL, 10);
}

int main(void) {
    if (!fahren_cuda_available()) {
        printf("SKIP: CUDA not available in this build\n");
        return 0;
    }

    const char* dir = getenv("FAHREN_MNIST_DIR");
    if (!dir || !*dir) dir = FAHREN_MNIST_DIR;

    FahrenEasyMnistPaths paths;
    if (fahren_easy_mnist_paths(dir, &paths) != FAHREN_SUCCESS) {
        printf("SKIP: MNIST not found in %s\n", dir);
        return 0;
    }

    float* train_x = NULL;
    int* train_y = NULL;
    size_t train_n = 0;
    float* test_x = NULL;
    int* test_y = NULL;
    size_t test_n = 0;

    FAHREN_ASSERT(fahren_mnist_load_dataset(paths.train_images, paths.train_labels,
                                            &train_x, &train_y, &train_n) == FAHREN_SUCCESS,
                  "load train");
    FAHREN_ASSERT(fahren_mnist_load_dataset(paths.test_images, paths.test_labels,
                                            &test_x, &test_y, &test_n) == FAHREN_SUCCESS,
                  "load test");

    size_t cap = env_size_t("FAHREN_MNIST_TRAIN_SAMPLES", 10000);
    if (cap > train_n) cap = train_n;
    size_t epochs = env_size_t("FAHREN_MNIST_EPOCHS", 5);

    remove(WEIGHTS);

    FAHRENModel* m = fahren_easy_model_dense(784, WEIGHTS,
        FAHREN_LAYER_ACTIVATION_RELU, 128,
        FAHREN_LAYER_ACTIVATION_RELU, 64,
        FAHREN_LAYER_ACTIVATION_SOFTMAX, 10,
        FAHREN_EASY_END);
    FAHREN_ASSERT(m != NULL, "build model");

    FAHRENTrainConfig cfg = fahren_train_config_cuda(0.01f);
    FAHREN_ASSERT(fahren_easy_train(m, WEIGHTS, train_x, train_y, cap, 784, 10, epochs, &cfg)
                  == FAHREN_SUCCESS, "train CUDA");

    float acc = 0.0f;
    FAHREN_ASSERT(fahren_easy_accuracy(m, WEIGHTS, test_x, test_y, test_n, 784, &acc) == FAHREN_SUCCESS,
                  "evaluate");
    printf("CUDA accuracy: %.2f%%\n", acc * 100.0f);

    fahren_mnist_free_dataset(train_x, train_y);
    fahren_mnist_free_dataset(test_x, test_y);
    fahren_shutdown(m);
    remove(WEIGHTS);

    FAHREN_ASSERT(acc >= 0.85f, "accuracy threshold");
    printf("\n%d passed, %d failed\n", g_fahren_tests_run - g_fahren_tests_failed, g_fahren_tests_failed);
    return g_fahren_tests_failed > 0 ? 1 : 0;
}
