from keras_custom.layers import Linear, MulConstant, PlusConstant
from keras.models import Sequential
from keras_custom.backward import get_backward
import numpy as np
from .conftest import linear_mapping, is_invertible, serialize


def test_backward_Linear():

    layer_0 = MulConstant(constant=1.0)
    layer_1 = PlusConstant(constant=0, minus=True)

    layer_linear = Linear([layer_0, layer_1])

    model_layer = Sequential([layer_linear])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer_linear, use_bias=False)
    linear_mapping(layer_linear, backward_layer)
    is_invertible(layer_linear, backward_layer)
    # use_bias should have no impact
    backward_layer = get_backward(layer_linear, use_bias=True)
    linear_mapping(layer_linear, backward_layer)
    is_invertible(layer_linear, backward_layer)
    serialize(layer_linear, backward_layer)
