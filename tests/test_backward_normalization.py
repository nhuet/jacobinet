from keras.layers import BatchNormalization
from keras.models import Sequential
from jacobinet import get_backward_layer as get_backward
import numpy as np
from .conftest import linear_mapping, is_invertible, serialize


def test_backward_BatchNormalization():

    layer = BatchNormalization()
    layer.trainable = False
    model_layer = Sequential([layer])
    model_layer.trainable = False
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)

    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)

    layer = BatchNormalization()
    layer.trainable=False
    model_layer = Sequential([layer])
    model_layer.trainable=False
    input_shape = (1,32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)

    layer = BatchNormalization()
    layer.trainable=False
    model_layer = Sequential([layer])
    model_layer.trainable=False
    input_shape = (1,32,32)
    _ = model_layer(np.ones(input_shape)[None])
    # use_bias should have an impact
    backward_layer = get_backward(layer)
    serialize(layer, backward_layer)
