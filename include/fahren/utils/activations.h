/*
 * FAHREN Activation Functions Utilities
 * 
 * Shared activation function implementations and derivatives.
 */

#ifndef FAHREN_ACTIVATIONS_H
#define FAHREN_ACTIVATIONS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward activation function
 * 
 * Parameters:
 *  - activation: activation type (FAHREN_LAYER_ACTIVATION_*)
 *  - x: input value
 * Returns: activated value
 */
float fahren_activation_forward(int activation, float x);

/* Activation derivative
 * 
 * Parameters:
 *  - activation: activation type
 *  - x: input value (original, not activated)
 * Returns: derivative value
 */
float fahren_activation_derivative(int activation, float x);

/* Batch forward activation */
void fahren_activation_forward_batch(int activation, const float* input,
                                    size_t count, float* output);

/* Batch activation derivative */
void fahren_activation_derivative_batch(int activation, const float* input,
                                       size_t count, float* output);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_ACTIVATIONS_H */
