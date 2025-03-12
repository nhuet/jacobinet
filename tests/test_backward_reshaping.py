from keras.layers import (
    Reshape,
    Flatten,
    Cropping2D,
    ZeroPadding2D,
    Cropping1D,
    ZeroPadding1D,
    Permute,
    RepeatVector,
    Cropping3D,
    UpSampling2D,
    UpSampling1D,
    UpSampling3D,
)
from keras.models import Sequential
from jacobinet import get_backward_layer as get_backward
import numpy as np
from .conftest import linear_mapping, serialize, is_invertible


def test_backward_Reshape():

    input_shape = (2, 5, 10)
    # data_format == 'channels_first'
    layer = Reshape((2, 50))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)


def test_backward_RepeatVector():

    input_shape = (10,)
    layer = RepeatVector(2)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)


def test_backward_Permute():

    input_shape = (2, 5, 10)
    # data_format == 'channels_first'
    layer = Permute((1, 3, 2))
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)

    serialize(layer, backward_layer)
    backward_layer = get_backward(layer, use_bias=True)
    linear_mapping(layer, backward_layer)


def test_backward_Flatten():

    input_shape = (2, 4, 3)
    # data_format == 'channels_first'
    layer = Flatten()
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    is_invertible(layer, backward_layer)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_Cropping2D():

    input_shape = (2, 12, 11)
    # data_format == 'channels_first'
    layer = Cropping2D(cropping=(3, 3), data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    input_shape = input_shape[::-1]
    layer = Cropping2D(cropping=(3, 3), data_format="channels_last")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_ZeroPadding2D():

    input_shape = (2, 12, 11)
    # data_format == 'channels_first'
    layer = ZeroPadding2D(padding=(3, 3), data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    input_shape = input_shape[::-1]
    layer = ZeroPadding2D(padding=(3, 3), data_format="channels_last")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_Cropping1D():

    input_shape = (11, 2)
    # data_format == 'channels_first'
    layer = Cropping1D(cropping=3)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_ZeroPadding1D():

    input_shape = (2, 12)
    # data_format == 'channels_first'
    layer = ZeroPadding1D(padding=3, data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    input_shape = input_shape[::-1]
    layer = ZeroPadding1D(padding=3, data_format="channels_last")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_Cropping3D():

    input_shape = (2, 12, 11, 10)
    # data_format == 'channels_first'
    layer = Cropping3D(cropping=(3, 3, 2), data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    input_shape = input_shape[::-1]
    layer = Cropping3D(cropping=(3, 3, 4), data_format="channels_last")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_UpSampling2D():

    input_shape = (2, 12, 11)
    # data_format == 'channels_first'
    layer = UpSampling2D(size=(2, 2), data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    input_shape = input_shape[::-1]
    layer = UpSampling2D(size=(2, 2), data_format="channels_last")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_UpSampling1D():

    input_shape = (2, 12)
    # data_format == 'channels_first'
    layer = UpSampling1D(size=3)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    input_shape = (2, 12)
    # data_format == 'channels_first'
    layer = UpSampling1D(size=2)
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


def test_backward_UpSampling3D():

    input_shape = (2, 12, 11, 10)

    # data_format == 'channels_first'
    layer = UpSampling3D(size=(3, 3, 2), data_format="channels_first")
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    backward_layer = get_backward(layer, use_bias=False)

    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)
