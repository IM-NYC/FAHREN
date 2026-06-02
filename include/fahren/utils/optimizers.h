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

typedef struct FAHRENOptimizerState FAHRENOptimizerState;

FAHRENOptimizer fahren_optimizer_create(FAHRENOptimizerType type, float lr);

FAHRENOptimizerState* fahren_optimizer_state_create(const FAHRENOptimizer* optimizer,
                                                    size_t param_count);
void fahren_optimizer_state_free(FAHRENOptimizerState* state);

void fahren_optimizer_update(FAHRENOptimizer* optimizer,
                             FAHRENOptimizerState* state,
                             float* params, const float* gradients,
                             size_t count, size_t iteration);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_OPTIMIZERS_H */
