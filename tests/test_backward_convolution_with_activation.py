import keras
import keras.ops as K
import numpy as np
import pytest
from jacobinet import clone_to_backward
from jacobinet import get_backward_layer as get_backward
from jacobinet.models import get_backward_sequential
from keras.layers import (
    Activation,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    Input,
    Reshape,
)
from keras.models import Sequential

from .conftest import compute_backward_model, linear_mapping, serialize, serialize_model


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_Conv3D(
    input_shape,
    filters,
    kernel_size,
    strides,
    padding,
    use_bias,
    activation="relu",
):
    # data_format == 'channels_first'
    layer = Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
        activation=activation,
    )

    input_dim = np.prod(input_shape)
    model = Sequential([Reshape(input_shape), layer, Reshape((-1,)), Dense(1)])
    _ = model(np.ones(input_dim)[None])
    backward_layer = get_backward(layer)  # check it is running
    backward_model = get_backward_sequential(model)
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_Conv2D(
    input_shape,
    filters,
    kernel_size,
    strides,
    padding,
    use_bias,
    activation="relu",
):
    # data_format == 'channels_first'
    layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
        activation=activation,
    )
    input_dim = np.prod(input_shape)
    model = Sequential([Reshape(input_shape), layer, Reshape((-1,)), Dense(1)])
    _ = model(np.ones(input_dim)[None])
    backward_layer = get_backward(layer)  # check it is running
    backward_model = get_backward_sequential(model)

    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


def _test_backward_Conv1D(
    input_shape,
    filters,
    kernel_size,
    strides,
    padding,
    use_bias,
    activation="relu",
):
    # data_format == 'channels_first'
    layer = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
        activation=activation,
    )
    input_dim = np.prod(input_shape)
    model = Sequential([Reshape(input_shape), layer, Reshape((-1,)), Dense(1)])
    _ = model(np.ones(input_dim)[None])
    backward_layer = get_backward(layer)  # check it is running
    backward_model = get_backward_sequential(model)
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_DepthwiseConv2D(
    input_shape,
    depth_multiplier,
    kernel_size,
    strides,
    padding,
    use_bias,
    activation="relu",
):
    # data_format == 'channels_first'
    layer = DepthwiseConv2D(
        depth_multiplier=depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
        activation=activation,
    )

    layer_wo_activation = DepthwiseConv2D(
        depth_multiplier=depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
    )

    input_dim = np.prod(input_shape)
    model = Sequential([Reshape(input_shape), layer, Reshape((-1,)), Dense(1)])
    model_split = Sequential(
        [
            Reshape(input_shape),
            layer_wo_activation,
            Activation(activation),
            Reshape((-1,)),
            Dense(1),
        ]
    )
    _ = model(np.ones(input_dim)[None])
    _ = model_split(np.ones(input_dim)[None])
    # same weights
    model_split.set_weights(model.get_weights())
    backward_layer = get_backward(layer)  # check it is running
    backward_model = clone_to_backward(model)

    backward_model_split = clone_to_backward(model_split)

    input_dim = np.prod(input_shape)
    batch = 32
    input = np.reshape(100 * (np.random.rand(batch * input_dim) - 0.5), [batch, input_dim])

    output_model = backward_model.predict([input, np.ones((batch, 1))])
    output_model_split = backward_model_split.predict([input, np.ones((batch, 1))])

    np.testing.assert_almost_equal(output_model, output_model_split, err_msg="corrupted weights")

    serialize_model([input_dim, 1], backward_model)


# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_DepthwiseConv1D(
    input_shape,
    depth_multiplier,
    kernel_size,
    strides,
    padding,
    use_bias,
    activation="relu",
):
    # data_format == 'channels_first'
    layer = DepthwiseConv1D(
        depth_multiplier=depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
        activation=activation,
    )
    layer_wo_activation = DepthwiseConv1D(
        depth_multiplier=depth_multiplier,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format="channels_first",
        use_bias=use_bias,
    )

    input_dim = np.prod(input_shape)
    model = Sequential([Reshape(input_shape), layer, Reshape((-1,)), Dense(1)])
    model_split = Sequential(
        [
            Reshape(input_shape),
            layer_wo_activation,
            Activation(activation),
            Reshape((-1,)),
            Dense(1),
        ]
    )
    _ = model(np.ones(input_dim)[None])
    _ = model_split(np.ones(input_dim)[None])
    # same weights
    model_split.set_weights(model.get_weights())
    backward_layer = get_backward(layer)  # check it is running
    backward_model = clone_to_backward(model)

    backward_model_split = clone_to_backward(model_split)

    input_dim = np.prod(input_shape)
    batch = 32
    input = np.reshape(100 * (np.random.rand(batch * input_dim) - 0.5), [batch, input_dim])

    output_model = backward_model.predict([input, np.ones((batch, 1))])
    output_model_split = backward_model_split.predict([input, np.ones((batch, 1))])

    np.testing.assert_almost_equal(output_model, output_model_split, err_msg="corrupted weights")

    serialize_model([input_dim, 1], backward_model)


def test_backward_DepthwiseConv2D():
    input_shape = (1, 32, 32)
    kernel_size = (2, 2)
    strides = (1, 1)
    depth_multiplier = 2
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )

    input_shape = (1, 31, 31)
    kernel_size = (2, 2)
    strides = (1, 1)
    depth_multiplier = 2
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )

    input_shape = (1, 32, 32)
    kernel_size = (2, 2)
    strides = (3, 2)
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )

    input_shape = (1, 32, 32)
    kernel_size = (4, 3)
    strides = (3, 2)
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv2D(
        input_shape, depth_multiplier, kernel_size, strides, padding, use_bias
    )


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_backward_DepthwiseConv1D(activation):
    input_shape = (10, 32)
    kernel_size = 2
    strides = 1
    depth_multiplier = 3
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (10, 31)
    kernel_size = 2
    strides = 1
    depth_multiplier = 2
    padding = "valid"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (1, 32)
    kernel_size = 2
    strides = 3
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (11, 32)
    kernel_size = 4
    strides = 3
    depth_multiplier = 2
    padding = "same"
    use_bias = False
    _test_backward_DepthwiseConv1D(
        input_shape,
        depth_multiplier,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_backward_Conv3D(activation):
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
    _test_backward_Conv3D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (1, 31, 31, 30)
    kernel_size = (2, 2, 1)
    strides = (1, 1, 1)
    filters = 2
    padding = "valid"
    use_bias = False
    _test_backward_Conv3D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (1, 32, 32, 31)
    kernel_size = (2, 2, 3)
    strides = (3, 2, 1)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv3D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (4, 32, 32, 30)
    kernel_size = (4, 3, 3)
    strides = (3, 2, 2)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv3D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_backward_Conv2D(activation):
    input_shape = (3, 32, 32)
    kernel_size = (2, 2)
    strides = (1, 1)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv2D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (1, 31, 31)
    kernel_size = (2, 2)
    strides = (1, 1)
    filters = 2
    padding = "valid"
    use_bias = False
    _test_backward_Conv2D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (1, 32, 32)
    kernel_size = (2, 2)
    strides = (3, 2)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv2D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (4, 32, 32)
    kernel_size = (4, 3)
    strides = (3, 2)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv2D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def Atest_backward_Conv1D(activation):
    input_shape = (3, 32)
    kernel_size = (2,)
    strides = 1
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv1D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (1, 31)
    kernel_size = (2,)
    strides = (1,)
    filters = 2
    padding = "valid"
    use_bias = False
    _test_backward_Conv1D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )

    input_shape = (
        4,
        32,
    )
    kernel_size = (4,)
    strides = (3,)
    filters = 2
    padding = "same"
    use_bias = False
    _test_backward_Conv1D(
        input_shape,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation=activation,
    )
