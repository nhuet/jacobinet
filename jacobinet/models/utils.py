import keras
from keras.layers import Layer, Input
from keras.models import Model
from jacobinet.layers.layer import (
    BackwardLayer,
    BackwardLinearLayer,
    BackwardNonLinearLayer,
)
from jacobinet.layers.convert import get_backward as get_backward_layer
from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List, Callable


def is_linear(model_backward: keras.models.Model) -> bool:
    return min(
        [
            isinstance(layer, BackwardLinearLayer)
            for layer in model_backward.layers
            if isinstance(layer, BackwardLayer)
        ]
    )


def get_backward(
    layer: Union[Layer, Model],
    gradient_shape=Union[None, Tuple[int], List[Tuple[int]]],
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    get_backward_model: Callable = None,
):
    if isinstance(layer, Layer):
        return get_backward_layer(layer, False, mapping_keras2backward_classes)
    else:

        raise NotImplementedError()
        if gradient_shape is None:
            return get_backward_model(
                layer,
                mapping_keras2backward_classes=mapping_keras2backward_classes,
            )
        else:
            gradient_shape = list(gradient_shape)
            gradients = [Input(shape_i) for shape_i in gradient_shape]
            return get_backward_model(
                layer,
                gradient=gradients,
                use_gradient_as_backward_input=True,
                mapping_keras2backward_classes=mapping_keras2backward_classes,
            )
