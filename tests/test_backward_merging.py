import keras
import numpy as np
import pytest
import torch
from jacobinet.models.model import get_backward_functional as get_backward_model
from jacobinet.models.sequential import get_backward_sequential
from keras.layers import (
    Add,
    Average,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Maximum,
    Minimum,
    Multiply,
    ReLU,
    Reshape,
    Subtract,
)
from keras.models import Model, Sequential

from .conftest import compute_backward_model, compute_output, serialize_model


def _test_model_multiD_multi_output(merge_layer, n_inputs, data_format):
    input_dim = 36
    input_ = Input((input_dim,))
    output_bones = []
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    for _ in range(n_inputs):
        layers_bone = [
            Reshape(target_shape),
            Conv2D(2, (3, 3)),
            ReLU(),
            Reshape((-1,)),
            Dense(10),
        ]
        output_bone = compute_output(input_, layers_bone)
        output_bones.append(output_bone)

    input_neck = merge_layer(output_bones)
    layers_neck = [ReLU(), Dense(5), ReLU(), Dense(4)]
    output = compute_output(input_neck, layers_neck)

    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    gradient = Input((4,))
    backward_model = get_backward_model(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 4))])

    for i in range(4):
        compute_backward_model((input_dim,), model, backward_model, i)

    serialize_model([input_dim, 4], backward_model)


def _test_model_multiD_multi_output_concat(merge_layer, n_inputs, axis, data_format):
    input_dim = 36
    input_ = Input((input_dim,))
    output_bones = []
    if data_format == "channels_first":
        target_shape = (1, 6, 6)
    else:
        target_shape = (6, 6, 1)
    for _ in range(n_inputs):
        layers_bone = [
            Reshape(target_shape),
            Conv2D(2, (3, 3)),
            ReLU(),
            Reshape((-1,)),
            Dense(10),
        ]
        if axis == -1:
            layers_bone.append(Reshape((10, 1)))
        else:
            layers_bone.append(Reshape((1, 10)))
        output_bone = compute_output(input_, layers_bone)
        output_bones.append(output_bone)

    input_neck = merge_layer(output_bones)
    layers_neck = [Flatten(), ReLU(), Dense(5), ReLU(), Dense(4)]
    output = compute_output(input_neck, layers_neck)

    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    gradient = Input((4,))
    backward_model = get_backward_model(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 4))])

    for i in range(4):
        compute_backward_model((input_dim,), model, backward_model, i)

    serialize_model([input_dim, 4], backward_model)


def test_backward_add():
    data_format = keras.config.image_data_format()
    keras.config.set_image_data_format(data_format)
    merge_layer = Add()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)

    merge_layer = Add()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)


def test_backward_subtract():
    data_format = keras.config.image_data_format()
    keras.config.set_image_data_format(data_format)
    merge_layer = Subtract()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)


def test_backward_average():
    data_format = keras.config.image_data_format()
    keras.config.set_image_data_format(data_format)
    merge_layer = Average()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)

    merge_layer = Average()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)


def _test_backward_maximum():
    data_format = keras.config.image_data_format()
    merge_layer = Maximum()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)

    merge_layer = Maximum()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_minimum(data_format):
    keras.config.set_image_data_format(data_format)
    merge_layer = Minimum()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)

    merge_layer = Minimum()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_backward_multiply(data_format):
    keras.config.set_image_data_format(data_format)
    merge_layer = Multiply()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs, data_format=data_format)


@pytest.mark.parametrize("axis", [-1, 1])
def test_backward_concatenate(axis):
    keras.config.set_image_data_format("channels_first")
    data_format = keras.config.image_data_format()
    merge_layer = Concatenate(axis=axis)
    n_inputs = 2
    _test_model_multiD_multi_output_concat(merge_layer, n_inputs, axis, data_format=data_format)

    merge_layer = Concatenate(axis=axis)
    n_inputs = 5
    _test_model_multiD_multi_output_concat(merge_layer, n_inputs, axis, data_format=data_format)
