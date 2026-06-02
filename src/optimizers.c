#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fahren/utils/optimizers.h>

struct FAHRENOptimizerState {
    FAHRENOptimizerType type;
    size_t count;
    size_t step;
    float* m;
    float* v;
    float* velocity;
};

FAHRENOptimizer fahren_optimizer_create(FAHRENOptimizerType type, float lr) {
    FAHRENOptimizer opt;
    memset(&opt, 0, sizeof(opt));
    opt.type = type;
    opt.learning_rate = lr;
    opt.momentum = 0.9f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.epsilon = 1e-8f;
    opt.decay = 0.9f;
    return opt;
}

FAHRENOptimizerState* fahren_optimizer_state_create(const FAHRENOptimizer* optimizer,
                                                    size_t param_count) {
    if (!optimizer || param_count == 0) return NULL;

    FAHRENOptimizerState* st = (FAHRENOptimizerState*)calloc(1, sizeof(*st));
    if (!st) return NULL;

    st->type = optimizer->type;
    st->count = param_count;
    st->step = 0;

    if (optimizer->type == FAHREN_OPTIMIZER_MOMENTUM) {
        st->velocity = (float*)calloc(param_count, sizeof(float));
        if (!st->velocity) { free(st); return NULL; }
    } else if (optimizer->type == FAHREN_OPTIMIZER_ADAM) {
        st->m = (float*)calloc(param_count, sizeof(float));
        st->v = (float*)calloc(param_count, sizeof(float));
        if (!st->m || !st->v) {
            free(st->m);
            free(st->v);
            free(st);
            return NULL;
        }
    } else if (optimizer->type == FAHREN_OPTIMIZER_RMSPROP) {
        st->v = (float*)calloc(param_count, sizeof(float));
        if (!st->v) { free(st); return NULL; }
    }

    return st;
}

void fahren_optimizer_state_free(FAHRENOptimizerState* state) {
    if (!state) return;
    free(state->m);
    free(state->v);
    free(state->velocity);
    free(state);
}

void fahren_optimizer_update(FAHRENOptimizer* optimizer,
                             FAHRENOptimizerState* state,
                             float* params, const float* gradients,
                             size_t count, size_t iteration) {
    if (!optimizer || !params || !gradients || count == 0) return;
    (void)iteration;

    const float lr = optimizer->learning_rate;
    size_t i;

    switch (optimizer->type) {
        case FAHREN_OPTIMIZER_SGD:
            for (i = 0; i < count; ++i) {
                params[i] -= lr * gradients[i];
            }
            break;

        case FAHREN_OPTIMIZER_MOMENTUM:
            if (!state || !state->velocity) {
                for (i = 0; i < count; ++i) params[i] -= lr * gradients[i];
                break;
            }
            for (i = 0; i < count; ++i) {
                state->velocity[i] = optimizer->momentum * state->velocity[i] - lr * gradients[i];
                params[i] += state->velocity[i];
            }
            break;

        case FAHREN_OPTIMIZER_ADAM:
            if (!state || !state->m || !state->v) {
                for (i = 0; i < count; ++i) params[i] -= lr * gradients[i];
                break;
            }
            state->step++;
            {
                float t = (float)state->step;
                float bc1 = 1.0f - powf(optimizer->beta1, t);
                float bc2 = 1.0f - powf(optimizer->beta2, t);
                if (bc1 < 1e-8f) bc1 = 1e-8f;
                if (bc2 < 1e-8f) bc2 = 1e-8f;
                for (i = 0; i < count; ++i) {
                    state->m[i] = optimizer->beta1 * state->m[i] + (1.0f - optimizer->beta1) * gradients[i];
                    state->v[i] = optimizer->beta2 * state->v[i] + (1.0f - optimizer->beta2) * gradients[i] * gradients[i];
                    float m_hat = state->m[i] / bc1;
                    float v_hat = state->v[i] / bc2;
                    params[i] -= lr * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
                }
            }
            break;

        case FAHREN_OPTIMIZER_RMSPROP:
            if (!state || !state->v) {
                for (i = 0; i < count; ++i) params[i] -= lr * gradients[i];
                break;
            }
            for (i = 0; i < count; ++i) {
                state->v[i] = optimizer->decay * state->v[i] + (1.0f - optimizer->decay) * gradients[i] * gradients[i];
                params[i] -= lr * gradients[i] / (sqrtf(state->v[i]) + optimizer->epsilon);
            }
            break;

        default:
            for (i = 0; i < count; ++i) params[i] -= lr * gradients[i];
            break;
    }
}
