/*
 * FAHREN Sequential Model API
 * 
 * Supports sequential stacking of layers with training via backpropagation.
 * Currently implemented layer types: DENSE
 * Planned layer types: CONVOLUTIONAL, POOLING
 */

#ifndef FAHREN_SEQUENTIAL_H
#define FAHREN_SEQUENTIAL_H

#include <fahren/fahren.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sequential model creation - currently the default model type */
FAHRENModel* fahren_create_sequential_model(int layer_count);

/* Add dense layer to sequential model */
void fahren_add_dense_layer(FAHRENModel* cm, int activation, int units);

/* Add convolutional layer (structure defined, training pending) */
void fahren_add_conv_layer(FAHRENModel* cm, int activation, 
                          int filters, int kernel_size, int stride);

/* Add pooling layer (structure defined, training pending) */
void fahren_add_pooling_layer(FAHRENModel* cm, int pool_size, int stride);

/* Train sequential model */
int fahren_train_sequential(FAHRENModel* cm, const float* inputs, size_t sample_count, 
                           size_t input_dim, const int* labels, size_t num_classes,
                           const char* weights_path, size_t epochs, float learning_rate);

/* Predict with sequential model */
int fahren_predict_sequential(FAHRENModel* cm, const float* input, 
                             size_t input_dim, float* output);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_SEQUENTIAL_H */
