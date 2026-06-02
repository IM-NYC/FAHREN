/*
 * FAHREN CLI runner — train / eval using ~/.fahren/config.ini
 *
 * Usage:
 *   fahren_cli train [config.ini]
 *   fahren_cli eval  [config.ini]
 *   fahren_cli status [config.ini]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fahren/fahren_easy.h>

#ifdef _WIN32
#include <direct.h>
#define FAHREN_HOME_DIR getenv("USERPROFILE")
#else
#include <unistd.h>
#include <pwd.h>
#define FAHREN_HOME_DIR (getenv("HOME"))
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
} FahrenCliConfig;

static void trim(char* s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) {
        s[--n] = '\0';
    }
}

static void default_config(FahrenCliConfig* c, const char* repo_hint) {
    memset(c, 0, sizeof(*c));
    strcpy(c->device, "cpu");
    c->epochs = 5;
    c->batch_size = 64;
    c->learning_rate = 0.01f;
    c->train_samples = 10000;
    c->min_accuracy = 0.85f;
    if (repo_hint) snprintf(c->mnist_dir, sizeof(c->mnist_dir), "%s/mnist", repo_hint);
    snprintf(c->weights_path, sizeof(c->weights_path), "fahren_model.bin");
    if (repo_hint) snprintf(c->build_dir, sizeof(c->build_dir), "%s/build", repo_hint);
}

static int load_config(const char* path, FahrenCliConfig* c) {
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
        trim(key);
        trim(val);
        if (strcmp(key, "device") == 0) snprintf(c->device, sizeof(c->device), "%s", val);
        else if (strcmp(key, "epochs") == 0) c->epochs = (size_t)strtoul(val, NULL, 10);
        else if (strcmp(key, "batch_size") == 0) c->batch_size = (size_t)strtoul(val, NULL, 10);
        else if (strcmp(key, "learning_rate") == 0) c->learning_rate = (float)strtod(val, NULL);
        else if (strcmp(key, "train_samples") == 0) c->train_samples = (size_t)strtoul(val, NULL, 10);
        else if (strcmp(key, "min_accuracy") == 0) c->min_accuracy = (float)strtod(val, NULL);
        else if (strcmp(key, "mnist_dir") == 0) snprintf(c->mnist_dir, sizeof(c->mnist_dir), "%s", val);
        else if (strcmp(key, "weights_path") == 0) snprintf(c->weights_path, sizeof(c->weights_path), "%s", val);
        else if (strcmp(key, "build_dir") == 0) snprintf(c->build_dir, sizeof(c->build_dir), "%s", val);
        else if (strcmp(key, "last_accuracy") == 0) { (void)val; }
    }
    fclose(f);
    return 0;
}

static void config_path_default(char* out, size_t len) {
    const char* home = FAHREN_HOME_DIR;
    if (!home) home = ".";
#ifdef _WIN32
    snprintf(out, len, "%s\\.fahren\\config.ini", home);
#else
    snprintf(out, len, "%s/.fahren/config.ini", home);
#endif
}

static int apply_device(const FahrenCliConfig* c) {
    if (strcmp(c->device, "cuda") == 0 || strcmp(c->device, "gpu") == 0) {
        if (!fahren_cuda_available()) {
            fprintf(stderr, "CUDA requested but not available. Build with -DFAHREN_ENABLE_CUDA=ON\n");
            return FAHREN_ERROR_UNSUPPORTED;
        }
        return fahren_set_device(FAHREN_DEVICE_CUDA);
    }
    return fahren_set_device(FAHREN_DEVICE_CPU);
}

static int cmd_status(const FahrenCliConfig* c) {
    FahrenEasyMnistPaths p;
    int mnist = (fahren_easy_mnist_paths(c->mnist_dir, &p) == FAHREN_SUCCESS);
    printf("device=%s cuda=%s mnist=%s weights=%s\n",
           c->device,
           fahren_cuda_available() ? "yes" : "no",
           mnist ? "ok" : "missing",
           c->weights_path);
    return 0;
}

static int cmd_train(const FahrenCliConfig* c) {
    int rc = apply_device(c);
    if (rc != FAHREN_SUCCESS) return rc;

    FahrenEasyMnistPaths paths;
    rc = fahren_easy_mnist_paths(c->mnist_dir, &paths);
    if (rc != FAHREN_SUCCESS) {
        fprintf(stderr, "MNIST not found in %s\n", c->mnist_dir);
        return rc;
    }

    float* train_x = NULL;
    int* train_y = NULL;
    size_t train_n = 0;

    rc = fahren_mnist_load_dataset(paths.train_images, paths.train_labels,
                                   &train_x, &train_y, &train_n);
    if (rc != FAHREN_SUCCESS) return rc;

    size_t cap = c->train_samples;
    if (cap > train_n) cap = train_n;

    remove(c->weights_path);

    FAHRENModel* m = fahren_easy_model_dense(784, c->weights_path,
        FAHREN_LAYER_ACTIVATION_RELU, 128,
        FAHREN_LAYER_ACTIVATION_RELU, 64,
        FAHREN_LAYER_ACTIVATION_SOFTMAX, 10,
        FAHREN_EASY_END);
    if (!m) {
        fahren_mnist_free_dataset(train_x, train_y);
        return FAHREN_ERROR_PROCESSING_FAILED;
    }

    FAHRENTrainConfig tcfg;
    if (strcmp(c->device, "cuda") == 0) {
        tcfg = fahren_train_config_cuda(c->learning_rate);
    } else {
        tcfg = fahren_train_config_default(c->learning_rate);
    }
    tcfg.batch_size = c->batch_size;

    printf("Training: device=%s samples=%zu epochs=%zu batch=%zu\n",
           c->device, cap, c->epochs, tcfg.batch_size);

    rc = fahren_easy_train(m, c->weights_path, train_x, train_y, cap, 784, 10, c->epochs, &tcfg);
    fahren_mnist_free_dataset(train_x, train_y);
    fahren_shutdown(m);

    if (rc == FAHREN_SUCCESS) {
        printf("Training complete -> %s\n", c->weights_path);
    }
    return rc;
}

static int cmd_eval(const FahrenCliConfig* c) {
    FahrenEasyMnistPaths paths;
    if (fahren_easy_mnist_paths(c->mnist_dir, &paths) != FAHREN_SUCCESS) {
        fprintf(stderr, "MNIST not found\n");
        return FAHREN_ERROR_IO;
    }

    float* test_x = NULL;
    int* test_y = NULL;
    size_t test_n = 0;
    int rc = fahren_mnist_load_dataset(paths.test_images, paths.test_labels,
                                       &test_x, &test_y, &test_n);
    if (rc != FAHREN_SUCCESS) return rc;

    FAHRENModel* m = fahren_easy_open_dense(784, c->weights_path,
        FAHREN_LAYER_ACTIVATION_RELU, 128,
        FAHREN_LAYER_ACTIVATION_RELU, 64,
        FAHREN_LAYER_ACTIVATION_SOFTMAX, 10,
        FAHREN_EASY_END);
    if (!m) return FAHREN_ERROR_PROCESSING_FAILED;

    float acc = 0.0f;
    rc = fahren_easy_accuracy(m, c->weights_path, test_x, test_y, test_n, 784, &acc);
    fahren_mnist_free_dataset(test_x, test_y);
    fahren_shutdown(m);

    if (rc == FAHREN_SUCCESS) {
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

    FahrenCliConfig cfg;
    default_config(&cfg, NULL);
    load_config(cfg_path, &cfg);

    if (strcmp(cmd, "train") == 0) return cmd_train(&cfg);
    if (strcmp(cmd, "eval") == 0) return cmd_eval(&cfg);
    if (strcmp(cmd, "status") == 0) return cmd_status(&cfg);

    fprintf(stderr, "Usage: %s {train|eval|status} [config.ini]\n", argv[0]);
    return 1;
}
