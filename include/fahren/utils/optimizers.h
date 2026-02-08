/*
 * FAHREN Optimization Algorithms
 * 
 * Support for multiple optimizer types: SGD, Momentum, Adam, RMSprop
 */

#ifndef FAHREN_OPTIMIZERS_H
#define FAHREN_OPTIMIZERS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Optimizer types */
typedef enum {
    FAHREN_OPTIMIZER_SGD,           /* Standard stochastic gradient descent */
    FAHREN_OPTIMIZER_MOMENTUM,      /* SGD with momentum */
    FAHREN_OPTIMIZER_ADAM,          /* Adaptive moment estimation */
    FAHREN_OPTIMIZER_RMSPROP        /* Root mean square propagation */
} FAHRENOptimizerType;

/* Optimizer configuration structure */
typedef struct {
    FAHRENOptimizerType type;
    float learning_rate;
    
    /* SGD + Momentum parameters */
    float momentum;
    
    /* Adam parameters */
    float beta1;        /* exponential decay rate for 1st moment */
    float beta2;        /* exponential decay rate for 2nd moment */
    float epsilon;      /* small constant for numerical stability */
    
    /* RMSprop parameters */
    float decay;        /* decay rate for historical gradients */
} FAHRENOptimizer;

/* Create optimizer with default parameters */
FAHRENOptimizer fahren_optimizer_create(FAHRENOptimizerType type, float lr);

/* Update parameters using optimizer
 * 
 * Parameters:
 *  - optimizer: optimizer configuration
 *  - params: parameter vector to update
 *  - gradients: gradient vector
 *  - count: number of parameters
 *  - iteration: current iteration (for learning rate scheduling)
 */
void fahren_optimizer_update(FAHRENOptimizer* optimizer,
                            float* params, const float* gradients,
                            size_t count, size_t iteration);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_OPTIMIZERS_H */
