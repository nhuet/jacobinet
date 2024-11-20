from keras_custom.layers import MulConstant, PlusConstant
from keras.models import Sequential
from keras_custom.backward import get_backward
import numpy as np
from .conftest import linear_mapping, is_invertible, serialize


def test_backward_MulConstant():

    layer = MulConstant(constant=1.0)
    model_layer = Sequential([layer])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    is_invertible(layer, backward_layer)
    # use_bias should have no impact
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = MulConstant(constant=1.0)
    model_layer = Sequential([layer])
    input_shape = (1, 32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    is_invertible(layer, backward_layer)
    # use_bias should have no impact
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = MulConstant(constant=1.0)
    model_layer = Sequential([layer])
    input_shape = (1, 32, 32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    is_invertible(layer, backward_layer)
    # use_bias should have no impact
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)


def test_backward_PlusConstant():

    layer = PlusConstant(constant=0, minus=True)
    model_layer = Sequential([layer])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = PlusConstant(constant=0, minus=False)
    model_layer = Sequential([layer])
    input_shape = (1, 32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = PlusConstant(constant=2, minus=True)
    model_layer = Sequential([layer])
    input_shape = (1, 32, 32)
    _ = model_layer(np.ones(input_shape)[None])
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)
