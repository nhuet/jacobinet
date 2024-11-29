from keras.layers import Dense, EinsumDense
from keras.models import Sequential
from jacobinet import get_backward_layer as get_backward
import numpy as np
from .conftest import linear_mapping, serialize


def test_backward_Dense():

    layer = Dense(units=3, use_bias=False)
    model_layer = Sequential([layer])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = Dense(units=3, use_bias=False)
    model_layer = Sequential([layer])
    input_shape = (2, 32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)

    layer_0 = Dense(units=5, use_bias=False)
    model_layer_0 = Sequential([layer_0])
    input_shape = (2, 3, 31)
    _ = model_layer_0(np.ones(input_shape)[None])
    backward_layer_0 = get_backward(layer_0)
    linear_mapping(layer_0, backward_layer_0)
    serialize(layer_0, backward_layer_0)


def _test_backwardEinsumDense():

    layer = EinsumDense("ab,bc->ac", output_shape=64, bias_axes="c")
    model_layer = Sequential([layer])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])

    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
