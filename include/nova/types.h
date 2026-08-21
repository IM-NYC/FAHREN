#ifndef NOVA_TYPES_H
#define NOVA_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOVA_VERSION_MAJOR                1
#define NOVA_VERSION_MINOR                0
#define NOVA_VERSION_PATCH                0

#define NOVA_MODEL_SEQUENTIAL             0
#define NOVA_MODEL_LSTM                   1

#define NOVA_LAYER_DENSE                  0
#define NOVA_LAYER_CONVOLUTIONAL          1
#define NOVA_LAYER_POOLING                2
#define NOVA_LAYER_SUBMODEL               3

#define NOVA_ACTIVATION_RELU              0
#define NOVA_ACTIVATION_SIGMOID           1
#define NOVA_ACTIVATION_TANH              2
#define NOVA_ACTIVATION_SOFTMAX           3
#define NOVA_ACTIVATION_LINEAR            4

#define NOVA_DEVICE_CPU                   0
#define NOVA_DEVICE_CUDA                  1
#define NOVA_DEVICE_ROCM                  2
#define NOVA_DEVICE_OPENCL                3
#define NOVA_DEVICE_VULKAN                4
#define NOVA_DEVICE_ONEAPI                5
#define NOVA_DEVICE_DEFAULT               (-1)

#define NOVA_PRECISION_FP32               0
#define NOVA_PRECISION_FP16               1
#define NOVA_PRECISION_INT8               2
#define NOVA_PRECISION_INT4               3

#define NOVA_OPTIMIZER_SGD                0
#define NOVA_OPTIMIZER_MOMENTUM           1
#define NOVA_OPTIMIZER_ADAM               2
#define NOVA_OPTIMIZER_RMSPROP            3

#define NOVA_FILE_MAGIC                   0x4E4F5641u
#define NOVA_FILE_VERSION                 1u

#define NOVA_MAX_LAYERS                   64
#define NOVA_MAX_NAME_LEN                 256
#define NOVA_MAX_DEVICE_NAME              32

typedef struct NOVALayer {
    int layer_type;
    int activation;
    int density;
    size_t input_size;
    size_t output_size;
    int param1;
    int param2;
    void* sub_model;
} NOVALayer;

typedef struct NOVALayerParams {
    float* weights;
    float* biases;
    float* grad_weights;
    float* grad_biases;
    void* opt_state_w;
    void* opt_state_b;
    size_t weight_count;
    size_t bias_count;
} NOVALayerParams;

typedef struct NOVAOptimizer {
    int type;
    float learning_rate;
    float momentum;
    float beta1;
    float beta2;
    float epsilon;
    float decay;
} NOVAOptimizer;

typedef struct NOVAOptimizerState NOVAOptimizerState;

typedef struct NOVATrainConfig {
    size_t batch_size;
    float learning_rate;
    NOVAOptimizer* optimizer;
    int device;
} NOVATrainConfig;

typedef struct NOVAModel NOVAModel;

#ifdef __cplusplus
}
#endif

#endif /* NOVA_TYPES_H */
