#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nova/nova.h>
#include <nova/mnist.h>
#include <nova/errors.h>

/* internal helpers from libnova */
extern NOVA_Status nova_path_join(char* buf, size_t sz, const char* dir, const char* file);
extern void nova_path_native(char* path);

#ifdef _WIN32
#define NOVA_HOME_DIR getenv("USERPROFILE")
#else
#include <unistd.h>
#include <pwd.h>
#define NOVA_HOME_DIR (getenv("HOME"))
#endif

typedef struct {
    char device[16];
    size_t epochs;
    size_t batch_size;
    float learning_rate;
    size_t train_samples;
    float min_accuracy;
    char mnist_dir[512];
    char weights_path[512];
    char build_dir[512];
} NovaCliConfig;

static void trim(char* s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' '))
        s[--n] = '\0';
}

static void default_config(NovaCliConfig* c, const char* repo_hint) {
    memset(c, 0, sizeof(*c));
    strcpy(c->device, "cpu");
    c->epochs = 5;
    c->batch_size = 64;
    c->learning_rate = 0.01f;
    c->train_samples = 10000;
    c->min_accuracy = 0.85f;
    if (repo_hint) nova_path_join(c->mnist_dir, sizeof(c->mnist_dir), repo_hint, "mnist");
    snprintf(c->weights_path, sizeof(c->weights_path), "nova_model.bin");
    if (repo_hint) nova_path_join(c->build_dir, sizeof(c->build_dir), repo_hint, "build");
}

static int load_config(const char* path, NovaCliConfig* c) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[768];
    while (fgets(line, sizeof(line), f)) {
        trim(line);
        if (!line[0] || line[0] == '#' || line[0] == ';') continue;
        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        char* key = line;
        char* val = eq + 1;
        trim(key); trim(val);
        if (strcmp(key, "device") == 0) snprintf(c->device, sizeof(c->device), "%s", val);
        else if (strcmp(key, "epochs") == 0) c->epochs = (size_t)strtoul(val, NULL, 10);
        else if (strcmp(key, "batch_size") == 0) c->batch_size = (size_t)strtoul(val, NULL, 10);
        else if (strcmp(key, "learning_rate") == 0) c->learning_rate = (float)strtod(val, NULL);
        else if (strcmp(key, "train_samples") == 0) c->train_samples = (size_t)strtoul(val, NULL, 10);
        else if (strcmp(key, "min_accuracy") == 0) c->min_accuracy = (float)strtod(val, NULL);
        else if (strcmp(key, "mnist_dir") == 0) snprintf(c->mnist_dir, sizeof(c->mnist_dir), "%s", val);
        else if (strcmp(key, "weights_path") == 0) snprintf(c->weights_path, sizeof(c->weights_path), "%s", val);
        else if (strcmp(key, "build_dir") == 0) snprintf(c->build_dir, sizeof(c->build_dir), "%s", val);
    }
    fclose(f);
    return 0;
}

static void config_path_default(char* out, size_t len) {
    const char* home = NOVA_HOME_DIR;
    if (!home) home = ".";
#ifdef _WIN32
    snprintf(out, len, "%s\\.nova\\config.ini", home);
#else
    snprintf(out, len, "%s/.nova/config.ini", home);
#endif
}

static int cmd_status(const NovaCliConfig* c) {
    printf("device=%s mnist=%s weights=%s\n",
           c->device,
           c->mnist_dir,
           c->weights_path);
    return 0;
}

static int cmd_train(const NovaCliConfig* c) {
    float* train_x = NULL;
    int* train_y = NULL;
    size_t train_n = 0;

    const char* mnist_dir = c->mnist_dir;
    char train_img_path[1024], train_lbl_path[1024];
    nova_path_join(train_img_path, sizeof(train_img_path), mnist_dir, "train-images-idx3-ubyte");
    nova_path_join(train_lbl_path, sizeof(train_lbl_path), mnist_dir, "train-labels-idx1-ubyte"); 

    NOVA_Status rc = nova_mnist_load(train_img_path, train_lbl_path, &train_x, &train_y, &train_n);
    if (rc != NOVA_SUCCESS) {
        char dbuf[512];
        snprintf(dbuf, sizeof(dbuf), "%s", c->mnist_dir);
        nova_path_native(dbuf);
        fprintf(stderr, "MNIST not found in %s\n", dbuf);
        return rc;
    }

    size_t cap = c->train_samples;
    if (cap > train_n) cap = train_n;

    remove(c->weights_path);

    NOVAModel* m = nova_model_create(NOVA_MODEL_SEQUENTIAL, 3);
    if (!m) { nova_mnist_free(train_x, train_y); return NOVA_ERROR_PROCESSING_FAILED; }
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 128);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 64);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_SOFTMAX, 10);

    rc = nova_model_finalize(m, c->weights_path, 784);
    if (rc != NOVA_SUCCESS) { nova_model_destroy(m); nova_mnist_free(train_x, train_y); return rc; }

    NOVATrainConfig cfg = nova_train_config_default(c->learning_rate);
    cfg.batch_size = c->batch_size;

    printf("Training: samples=%zu epochs=%zu batch=%zu\n",
           cap, c->epochs, cfg.batch_size);

    rc = nova_train_with_config(m, train_x, cap, 784, train_y, 10, c->weights_path, c->epochs, &cfg);
    nova_mnist_free(train_x, train_y);
    nova_model_destroy(m);

    if (rc == NOVA_SUCCESS)
        printf("Training complete -> %s\n", c->weights_path);
    return rc;
}

static int cmd_eval(const NovaCliConfig* c) {
    float* test_x = NULL;
    int* test_y = NULL;
    size_t test_n = 0;

    const char* mnist_dir = c->mnist_dir;

    char test_img_path[1024], test_lbl_path[1024];
    nova_path_join(test_img_path, sizeof(test_img_path), mnist_dir, "t10k-images-idx3-ubyte");
    nova_path_join(test_lbl_path, sizeof(test_lbl_path), mnist_dir, "t10k-labels-idx1-ubyte");

    NOVA_Status rc = nova_mnist_load(test_img_path, test_lbl_path, &test_x, &test_y, &test_n);
    if (rc != NOVA_SUCCESS) {
        char dbuf[512];
        snprintf(dbuf, sizeof(dbuf), "%s", c->mnist_dir);
        nova_path_native(dbuf);
        fprintf(stderr, "MNIST test set not found in %s\n", dbuf);
        return rc;
    }

    NOVAModel* m = nova_model_create(NOVA_MODEL_SEQUENTIAL, 3);
    if (!m) { nova_mnist_free(test_x, test_y); return NOVA_ERROR_PROCESSING_FAILED; }
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 128);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_RELU, 64);
    nova_model_add_layer(m, NOVA_LAYER_DENSE, NOVA_ACTIVATION_SOFTMAX, 10);
    rc = nova_model_finalize(m, c->weights_path, 784);
    if (rc != NOVA_SUCCESS) { nova_model_destroy(m); nova_mnist_free(test_x, test_y); return rc; }

    float acc = 0.0f;
    rc = nova_evaluate(m, c->weights_path, test_x, test_y, test_n, 784, &acc);
    nova_mnist_free(test_x, test_y);
    nova_model_destroy(m);

    if (rc == NOVA_SUCCESS) {
        printf("accuracy=%.2f%%\n", acc * 100.0f);
        if (acc < c->min_accuracy) {
            fprintf(stderr, "Below min_accuracy %.0f%%\n", c->min_accuracy * 100.0f);
            return 1;
        }
    }
    return rc;
}

int main(int argc, char** argv) {
    const char* cmd = (argc > 1) ? argv[1] : "status";
    char cfg_path[512];
    config_path_default(cfg_path, sizeof(cfg_path));
    if (argc > 2) snprintf(cfg_path, sizeof(cfg_path), "%s", argv[2]);

    NovaCliConfig cfg;
    default_config(&cfg, NULL);
    load_config(cfg_path, &cfg);

    if (strcmp(cmd, "train") == 0) return cmd_train(&cfg);
    if (strcmp(cmd, "eval") == 0) return cmd_eval(&cfg);
    if (strcmp(cmd, "status") == 0) return cmd_status(&cfg);

    fprintf(stderr, "Usage: %s {train|eval|status} [config.ini]\n", argv[0]);
    return 1;
}
