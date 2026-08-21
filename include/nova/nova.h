#ifndef NOVA_H
#define NOVA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <nova/errors.h>
#include <nova/types.h>

NOVAModel* nova_model_create(int model_type, int layer_count);
void nova_model_destroy(NOVAModel* model);

void nova_model_add_layer(NOVAModel* model, int layer_type, int activation, ...);
NOVA_Status nova_model_finalize(NOVAModel* model, const char* path, int input_dim);

NOVA_Status nova_train(NOVAModel* model, const float* inputs, size_t sample_count,
                       size_t input_dim, const int* labels, size_t num_classes,
                       const char* path, size_t epochs, float learning_rate);
NOVA_Status nova_train_with_config(NOVAModel* model, const float* inputs,
                                   size_t sample_count, size_t input_dim,
                                   const int* labels, size_t num_classes,
                                   const char* path, size_t epochs,
                                   const NOVATrainConfig* config);
NOVATrainConfig nova_train_config_default(float learning_rate);

NOVA_Status nova_predict(NOVAModel* model, const char* path,
                         const float* input, size_t input_dim, int* class_out);
NOVA_Status nova_evaluate(NOVAModel* model, const char* path,
                          const float* inputs, const int* labels,
                          size_t sample_count, size_t input_dim, float* accuracy);

NOVA_Status nova_device_set(int device);
int nova_device_get(void);
int nova_device_available(int device);
const char* nova_device_name(int device);

NOVA_Status nova_save(NOVAModel* model, const char* dirpath);
NOVA_Status nova_load(NOVAModel* model, const char* dirpath);
NOVA_Status nova_verify(const char* dirpath);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_H */
