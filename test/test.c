#include <string.h>

#include <fahren/fahren.h>
#include <fahren/errors.h>
#include "test_harness.h"

#define WEIGHTS "fahren_smoke.bin"

static void test_errors(void) {
    FAHREN_ASSERT(strcmp(fahren_strerror(FAHREN_SUCCESS), "success") == 0, "strerror success");
    fahren_set_last_error("detail");
    FAHREN_ASSERT(strcmp(fahren_last_error_message(), "detail") == 0, "last error");
    fahren_clear_last_error();
}

static void test_tiny_train(void) {
    float x[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    int y[] = {0, 1, 0, 1};
    remove(WEIGHTS);

    FAHRENModel* m = fahren_create_model(FAHREN_MODEL_SEQUENTIAL, 2);
    FAHREN_ASSERT(m != NULL, "create model");
    fahren_add_layer(m, FAHREN_LAYER_DENSE, FAHREN_LAYER_ACTIVATION_RELU, 4);
    fahren_add_layer(m, FAHREN_LAYER_DENSE, FAHREN_LAYER_ACTIVATION_SOFTMAX, 2);
    FAHREN_ASSERT(fahren_finalize_model_to_file(m, WEIGHTS, 4) == FAHREN_SUCCESS, "finalize");

    FAHRENTrainConfig cfg = fahren_train_config_default(0.05f);
    cfg.batch_size = 2;
    FAHREN_ASSERT(fahren_train_with_config(m, x, 4, 4, y, 2, WEIGHTS, 2, &cfg) == FAHREN_SUCCESS, "train");

    fahren_shutdown(m);
    remove(WEIGHTS);
}

int main(void) {
    printf("FAHREN smoke (v%d.%d.%d)\n\n",
           FAHREN_VERSION_MAJOR, FAHREN_VERSION_MINOR, FAHREN_VERSION_PATCH);
    test_errors();
    test_tiny_train();
    printf("\n%d passed, %d failed\n", g_fahren_tests_run - g_fahren_tests_failed, g_fahren_tests_failed);
    return g_fahren_tests_failed > 0 ? 1 : 0;
}
