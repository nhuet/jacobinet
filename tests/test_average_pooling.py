import keras
import numpy as np
import pytest
import torch
from jacobinet import get_backward_layer as get_backward
from jacobinet.models import get_backward_sequential

# comparison with depthwiseConv for AveragePooling
from keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Reshape,
)
from keras.models import Model, Sequential

from .conftest import (
    compute_backward_layer,
    compute_backward_model,
    linear_mapping,
    serialize,
    serialize_model,
)


####### AveragePooling2D #######
# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding):
    # data_format == 'channels_first'
    layer = AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    # equivalent DepthwiseConv2D
    pool_size = list(layer.pool_size)
    layer_conv = DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=layer.pool_size,
        strides=layer.strides,
        padding=layer.padding,
        data_format=layer.data_format,
        use_bias=False,
        trainable=False,
    )
    kernel_ = np.ones(pool_size + [1, 1]) / np.prod(pool_size)
    # layer_conv.weights = [keras.Variable(kernel_)]
    layer_conv(layer.input)
    layer_conv.built = True
    layer_conv.set_weights([kernel_])

    # check equality
    if padding == "valid":
        random_input = np.reshape(np.random.rand(np.prod(input_shape) * 5), [5] + list(input_shape))
        output_pooling = layer(random_input)
        output_conv = layer_conv(random_input)
        np.testing.assert_almost_equal(
            output_pooling.cpu().numpy(), output_conv.cpu().numpy(), decimal=5
        )

    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)

    # data_format == 'channels_last'

    input_shape = input_shape[::-1]
    layer = AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format="channels_last",
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)

    # check gradient
    mask_output = Input(input_shape)
    input_ = Input(input_shape)
    output = backward_layer([mask_output, input_])
    model_backward = Model([mask_output, input_], output)

    compute_backward_layer(input_shape, model_layer, model_backward)


def test_backward_AveragePooling2D():
    pool_size = (2, 2)
    strides = (1, 1)
    padding = "valid"
    input_shape = (1, 32, 32)
    _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding)

    pool_size = (3, 3)
    strides = (2, 1)
    padding = "valid"
    input_shape = (1, 31, 32)
    _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding)

    # not working: same NotImplementedError
    pool_size = (3, 3)
    strides = (2, 2)
    padding = "valid"
    input_shape = (1, 31, 32)
    _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding)
