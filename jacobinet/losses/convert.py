from keras.losses import CategoricalCrossentropy

from .loss import Loss_Layer, CategoricalCrossentropy_Layer
from jacobinet.losses import BackwardLoss
from .categorical_crossentropy import get_backward_CategoricalCrossentropy

from typing import Optional

default_mapping_keras2backward_loss: dict[type[Loss_Layer], type[callable]] = {
    # convolution
    CategoricalCrossentropy_Layer: get_backward_CategoricalCrossentropy,
}

def get_backward_loss(
    layer: Loss_Layer,
    mapping_keras2backward_losses: Optional[
        dict[type[Loss_Layer], type[BackwardLoss]]
    ] = None,
    **kwargs,
):
    keras_class = type(layer)

    if mapping_keras2backward_losses is not None:
        default_mapping_keras2backward_loss.update(
            mapping_keras2backward_losses
        )

    get_backward_layer_loss = default_mapping_keras2backward_loss.get(keras_class)
    if get_backward_layer_loss is None:
        raise ValueError(
            "The backward mapping from the current loss is not native and not available in mapping_keras2backward_loss, {} not found".format(
                keras_class
            )
        )
    return get_backward_layer_loss(layer)