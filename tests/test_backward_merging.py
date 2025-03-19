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


def _test_model_multiD_multi_output(merge_layer, n_inputs):
    input_dim = 36
    input_ = Input((input_dim,))
    output_bones = []
    for _ in range(n_inputs):
        layers_bone = [
            Reshape((1, 6, 6)),
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


def _test_model_multiD_multi_output_concat(merge_layer, n_inputs, axis):
    input_dim = 36
    input_ = Input((input_dim,))
    output_bones = []
    for _ in range(n_inputs):
        layers_bone = [
            Reshape((1, 6, 6)),
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
    merge_layer = Add()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)

    merge_layer = Add()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs)


def test_backward_subtract():
    merge_layer = Subtract()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)


def test_backward_average():
    merge_layer = Average()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)

    merge_layer = Average()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs)


def _test_backward_maximum():
    merge_layer = Maximum()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)

    merge_layer = Maximum()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs)


def test_backward_minimum():
    merge_layer = Minimum()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)

    merge_layer = Minimum()
    n_inputs = 5
    _test_model_multiD_multi_output(merge_layer, n_inputs)


def test_backward_multiply():
    merge_layer = Multiply()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)


@pytest.mark.parametrize("axis", [-1, 1])
def test_backward_concatenate(axis):
    merge_layer = Concatenate(axis=axis)
    n_inputs = 2
    _test_model_multiD_multi_output_concat(merge_layer, n_inputs, axis)

    merge_layer = Concatenate(axis=axis)
    n_inputs = 5
    _test_model_multiD_multi_output_concat(merge_layer, n_inputs, axis)
