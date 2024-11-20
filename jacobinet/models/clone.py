from keras.models import Model, Sequential
from .sequential import get_backward_sequential
from .model import get_backward_model

from jacobinet.layers.layer import BackwardLayer
from typing import Optional


def clone_2_backward(
    model: Model,
    gradients=None,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
):
    if isinstance(model, Sequential):
        return get_backward_sequential(
            model, gradients, mapping_keras2backward_classes
        )
    else:
        return get_backward_model(
            model, gradients, mapping_keras2backward_classes
        )
