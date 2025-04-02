import keras
import numpy as np
import pytest
from jacobinet import get_backward_layer as get_backward
from keras.layers import BatchNormalization  # type:ignore
from keras.models import Sequential  # type:ignore

from .conftest import is_invertible, linear_mapping, serialize


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_BatchNormalization(data_format):
    keras.config.set_image_data_format(data_format)
    layer = BatchNormalization()
    layer.trainable = False

    model_layer = Sequential([layer])
    model_layer.trainable = False
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])

    weights = layer.get_weights()
    gamma = weights[0]
    gamma[0] = -1
    gamma[-1] = -2
    weights[0] = gamma
    layer.set_weights(weights)
    backward_layer = get_backward(layer)

    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)

    layer = BatchNormalization()
    layer.trainable = False
    model_layer = Sequential([layer])
    model_layer.trainable = False
    input_shape = (1, 32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)

    layer = BatchNormalization()
    layer.trainable = False
    model_layer = Sequential([layer])
    model_layer.trainable = False
    input_shape = (1, 32, 32)
    _ = model_layer(np.ones(input_shape)[None])
    # use_bias should have an impact
    backward_layer = get_backward(layer)
    serialize(layer, backward_layer)
