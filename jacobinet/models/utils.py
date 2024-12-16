import keras
from keras.layers import Layer, Input
from keras.models import Model, Sequential
from jacobinet.layers.layer import (
    BackwardLayer,
    BackwardLinearLayer,
    BackwardNonLinearLayer,
)
from jacobinet.layers.convert import get_backward as get_backward_layer
from keras.layers import Layer, InputLayer

from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List, Callable

def to_list(tensor:Union[Tensor, List[Tensor]])->List[Tensor]:
        if isinstance(tensor, list):
            return tensor
        return [tensor]

def is_linear(model_backward: keras.models.Model) -> bool:
    return hasattr(model_backward, 'is_linear') and model_backward.is_linear

def is_linear_layer(layer):
    if not (isinstance(layer, BackwardLayer) or (hasattr(layer, 'is_linear'))):
        return True
    return isinstance(layer, BackwardLinearLayer) or (hasattr(layer, 'is_linear') and layer.is_linear)

def get_backward(
    layer: Union[Layer, Model],
    gradient_shape=Union[None, Tuple[int], List[Tuple[int]]],
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    get_backward_model: Callable = None,
):
    if isinstance(layer, Layer):
        return get_backward_layer(layer, mapping_keras2backward_classes)
    else:

        raise NotImplementedError()

class GradConstant(Layer):

    def __init__(self, gradient, **kwargs):
        """
        to fill
        """
        super(GradConstant, self).__init__(**kwargs)
        self.grad_const = keras.ops.convert_to_tensor(gradient)

    def call(self, inputs_):
        return self.grad_const

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grad_const": keras.saving.serialize_keras_object(
                    self.grad_const
                ),
            }
        )
        return config

    def compute_output_shape(self, input_shape):

        return list(self.grad_const.shape)

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("grad_const")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)


def get_gradient(grad: Tensor, input) -> Tuple[Any, bool]:
    # if grad is an InputLayer return grad
    if isinstance(grad, InputLayer) or isinstance(grad, keras.KerasTensor):
        return grad  # grad is a KerasTensor that come from input or extra_inputs or is an Input Tensor
    # else return it as a Constant of a layer
    constant = GradConstant(gradient=grad)(input)
    return constant

class FuseGradients(Layer):

    def call(self, inputs, training=None, mask=None):
        # expand
        output = [K.expand_dims(input_, -1) for input_ in inputs]
        # concat
        output = K.concatenate(output, axis=-1)
        # sum
        return K.sum(output, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]