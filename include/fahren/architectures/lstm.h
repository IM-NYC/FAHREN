/*
 * FAHREN LSTM Model API
 * 
 * Long Short-Term Memory (LSTM) layers for sequence processing.
 * Currently defined but training not yet implemented.
 */

#ifndef FAHREN_LSTM_H
#define FAHREN_LSTM_H

#include <fahren/fahren.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LSTM model creation */
FAHRENModel* fahren_create_lstm_model(int layer_count);

/* Add LSTM layer
 * 
 * Parameters:
 *  - cm: model pointer
 *  - activation: activation function for cell state
 *  - units: number of LSTM units (hidden dimension)
 *  - return_sequences: if 1, return full sequence; if 0, return last output
 */
void fahren_add_lstm_layer(FAHRENModel* cm, int activation, 
                          int units, int return_sequences);

/* Train LSTM model on sequences
 * 
 * Input format: flattened sequence data [num_samples * seq_length * features]
 * Labels: output labels [num_samples]
 */
int fahren_train_lstm(FAHRENModel* cm, const float* sequences, 
                     size_t num_samples, size_t seq_length, size_t feature_dim,
                     const int* labels, size_t num_classes,
                     const char* weights_path, size_t epochs, float learning_rate);

/* Predict with LSTM model on a sequence */
int fahren_predict_lstm(FAHRENModel* cm, const float* sequence,
                       size_t seq_length, size_t feature_dim, float* output);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_LSTM_H */
