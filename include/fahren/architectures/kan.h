/*
 * FAHREN Kolmogorov-Arnold Network (KAN) API
 * 
 * Implements neural networks based on the Kolmogorov-Arnold representation theorem.
 * Each layer computes sums of learnable univariate functions (splines).
 * 
 * References:
 * - "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
 * - Kolmogorov-Arnold representation theorem (1957)
 */

#ifndef FAHREN_KAN_H
#define FAHREN_KAN_H

#include <fahren/fahren.h>

#ifdef __cplusplus
extern "C" {
#endif

/* KAN model creation */
FAHRENModel* fahren_create_kan_model(int layer_count);

/* Define KAN layer
 * 
 * Parameters:
 *  - cm: model pointer
 *  - activation: activation function to apply after spline sum
 *  - output_dim: output dimension
 *  - num_splines: number of B-spline basis functions per spline
 *  - spline_degree: degree of B-splines (1=linear, 2=quadratic, 3=cubic)
 */
void fahren_add_kan_layer(FAHRENModel* cm, int activation, 
                         int output_dim, int num_splines, int spline_degree);

/* Train KAN model */
int fahren_train_kan(FAHRENModel* cm, const float* inputs, size_t sample_count,
                    size_t input_dim, const int* labels, size_t num_classes,
                    const char* weights_path, size_t epochs, float learning_rate);

/* Predict with KAN model */
int fahren_predict_kan(FAHRENModel* cm, const float* input,
                      size_t input_dim, float* output);

/* Query learned spline functions (for interpretability)
 * Returns the spline coefficients for a specific connection
 */
int fahren_kan_get_spline_coeffs(FAHRENModel* cm, int layer_idx,
                                int output_idx, int input_idx,
                                float** coeffs_out, int* num_coeffs_out);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_KAN_H */
