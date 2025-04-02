import keras
import numpy as np
import pytest
from jacobinet import get_backward_layer as get_backward
from keras.layers import Conv1D, Conv2D, Conv3D, DepthwiseConv1D, DepthwiseConv2D
from keras.models import Sequential

from .conftest import linear_mapping, serialize


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_Conv3D(input_shape, filters, kernel_size, strides, padding, use_bias):
    # data_format == 'channels_first'
    layer = Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_Conv2D(input_shape, filters, kernel_size, strides, padding, use_bias):
    # data_format == 'channels_first'
    layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)


def _test_backward_Conv1D(input_shape, filters, kernel_size, strides, padding, use_bias):
    # data_format == 'channels_first'
    layer = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_DepthwiseConv2D(
    input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
):
    # data_format == 'channels_first'
    layer = DepthwiseConv2D(
        depth_multiplier=depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_DepthwiseConv1D(
    input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
):
    # data_format == 'channels_first'
    layer = DepthwiseConv1D(
        depth_multiplier=depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    serialize(layer, backward_layer)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_DepthwiseConv2D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (1, 32, 32)
    else:
        input_shape = (32, 32, 1)
    kernel_size = (2, 2)
    strides = (1, 1)
    depth_multiplier = 2
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )

    if data_format == "channels_first":
        input_shape = (1, 31, 31)
    else:
        input_shape = (31, 31, 1)
    kernel_size = (2, 2)
    strides = (1, 1)
    depth_multiplier = 2
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )
    if data_format == "channels_first":
        input_shape = (1, 32, 32)
    else:
        input_shape = (32, 32, 1)
    kernel_size = (2, 2)
    strides = (3, 2)
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )
    if data_format == "channels_first":
        input_shape = (1, 32, 32)
    else:
        input_shape = (32, 32, 1)
    kernel_size = (4, 3)
    strides = (3, 2)
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_DepthwiseConv1D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (10, 32)
    else:
        input_shape = (32, 10)
    kernel_size = 2
    strides = 1
    depth_multiplier = 3
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )
    if data_format == "channels_first":
        input_shape = (10, 31)
    else:
        input_shape = (31, 10)
    kernel_size = 2
    strides = 1
    depth_multiplier = 2
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )
    if data_format == "channels_first":
        input_shape = (1, 32)
    else:
        input_shape = (32, 1)
    kernel_size = 2
    strides = 3
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )
    if data_format == "channels_first":
        input_shape = (11, 32)
    else:
        input_shape = (32, 11)
    kernel_size = 4
    strides = 3
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )


def test_backward_Conv3D():
    keras.config.set_image_data_format("channels_first")
    pytest.skip("skip tests for 3D")
    # skip tests on MPS device as Conv3DTranspose is not implemented
    if keras.config.backend() == "torch":
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("skip tests on MPS device as Conv3DTranspose is not implemented")

    input_shape = (3, 32, 32, 31)
    kernel_size = (2, 2, 1)
    strides = (1, 1, 1)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv3D(input_shape, filters, kernel_size, strides, padding, use_bias)

    input_shape = (1, 31, 31, 30)
    kernel_size = (2, 2, 1)
    strides = (1, 1, 1)
    filters = 2
    padding = "valid"
    use_bias = False
    _test_backward_Conv3D(input_shape, filters, kernel_size, strides, padding, use_bias)

    input_shape = (1, 32, 32, 31)
    kernel_size = (2, 2, 3)
    strides = (3, 2, 1)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv3D(input_shape, filters, kernel_size, strides, padding, use_bias)

    input_shape = (4, 32, 32, 30)
    kernel_size = (4, 3, 3)
    strides = (3, 2, 2)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv3D(input_shape, filters, kernel_size, strides, padding, use_bias)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_Conv2D(data_format):
    keras.config.set_image_data_format(data_format)

    if data_format == "channels_first":
        input_shape = (3, 32, 32)
    else:
        input_shape = (32, 32, 3)
    kernel_size = (2, 2)
    strides = (1, 1)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv2D(input_shape, filters, kernel_size, strides, padding, use_bias)

    if data_format == "channels_first":
        input_shape = (1, 31, 31)
    else:
        input_shape = (31, 31, 1)
    kernel_size = (2, 2)
    strides = (1, 1)
    filters = 2
    padding = "valid"
    use_bias = False
    _test_backward_Conv2D(input_shape, filters, kernel_size, strides, padding, use_bias)

    if data_format == "channels_first":
        input_shape = (1, 32, 32)
    else:
        input_shape = (32, 32, 1)
    kernel_size = (2, 2)
    strides = (3, 2)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv2D(input_shape, filters, kernel_size, strides, padding, use_bias)

    if data_format == "channels_first":
        input_shape = (4, 32, 32)
    else:
        input_shape = (32, 32, 4)
    kernel_size = (4, 3)
    strides = (3, 2)
    filters = 1
    padding = "same"
    use_bias = False
    _test_backward_Conv2D(input_shape, filters, kernel_size, strides, padding, use_bias)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_Conv1D(data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (3, 32)
    else:
        input_shape = (32, 3)
    kernel_size = (2,)
    strides = 1
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv1D(input_shape, filters, kernel_size, strides, padding, use_bias)

    if data_format == "channels_first":
        input_shape = (1, 31)
    else:
        input_shape = (31, 1)
    kernel_size = (2,)
    strides = (1,)
    filters = 2
    padding = "valid"
    use_bias = False
    _test_backward_Conv1D(input_shape, filters, kernel_size, strides, padding, use_bias)

    if data_format == "channels_first":
        input_shape = (4, 32)
    else:
        input_shape = (32, 4)
    kernel_size = (4,)
    strides = (3,)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv1D(input_shape, filters, kernel_size, strides, padding, use_bias)
