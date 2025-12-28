#include <stdio.h>
#include <stdlib.h>
#include "../include/fahren/fahren.h"

int main(void) {
    FAHRENModel* sub_model = fahren_create_model(FAHREN_MODEL_SEQUENTIAL, 3);
    fahren_add_layer(sub_model, FAHREN_LAYER_DENSE, 20, NULL);
    fahren_add_layer(sub_model, FAHREN_LAYER_DENSE, 15, NULL);
    fahren_add_layer(sub_model, FAHREN_LAYER_DENSE, 10, NULL);

    // Create layers
    FAHRENModel* model = fahren_create_model(FAHREN_MODEL_SEQUENTIAL, 3);
    fahren_add_layer(model, FAHREN_LAYER_DENSE, 10, NULL);
    fahren_add_layer(model, FAHREN_LAYER_SUBMODEL, sub_model);
    fahren_add_layer(model, FAHREN_LAYER_DENSE, 5, NULL);

    // Use model...
    fahren_shutdown(model);
    fahren_shutdown(sub_model);

    return 0;
}