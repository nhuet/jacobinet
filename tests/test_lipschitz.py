# compare the output of the lipschitz model with the lipschitz function evaluated using torch backward

import keras
import numpy as np
import pytest
import torch
from jacobinet import get_lipschitz_model

# from jacobinet.models.sequential import get_backward_sequential
# from jacobinet.models.model import get_backward_functional
from jacobinet.models import clone_to_backward
from keras.layers import Dense, Input, ReLU
from keras.models import Model, Sequential


def _test_lipschitz_linear(p, model):
    backward_model = clone_to_backward(model)

    lipschitz_model = get_lipschitz_model(backward_model, p=p)
    lip_constant = lipschitz_model.predict(np.ones((1, 1)))

    # compute gradient by hand
    weights = model.get_weights()
    gradient = np.dot(weights[0], weights[1])[:, 0]

    if p == 2:
        lip_constant_np = np.sqrt(np.sum(gradient**2))
    elif p == np.inf:
        lip_constant_np = max(np.abs(gradient))
    elif p == 1:
        lip_constant_np = np.sum(np.abs(gradient))
    np.testing.assert_almost_equal(lip_constant, lip_constant_np, decimal=3)


def _test_lipschitz_non_linear(p, model, input_dim):
    backward_model = clone_to_backward(model)

    lipschitz_model = get_lipschitz_model(backward_model, p=p)

    random_input = np.asarray(np.random.rand(1, input_dim), dtype="float32")
    lip_constant = lipschitz_model.predict([random_input, np.ones((1, 1))])

    # compute gradient by hand

    input_torch = torch.tensor(random_input, requires_grad=True)
    output = model(input_torch)
    select_output = output[0, 0]
    select_output.backward()
    gradient = input_torch.grad.cpu().detach().numpy()[0]

    if p == 2:
        lip_constant_np = np.sqrt(np.sum(gradient**2))
    elif p == np.inf:
        lip_constant_np = max(np.abs(gradient))
    elif p == 1:
        lip_constant_np = np.sum(np.abs(gradient))

    np.testing.assert_almost_equal(lip_constant, lip_constant_np, decimal=3)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_lipschitz_sequential_linear(p):
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    _test_lipschitz_linear(p, model)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_lipschitz_model_linear(p):
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    input = Input((input_dim,))
    output = layers[1](layers[0](input))
    model = Model(input, output)
    _ = model(torch.ones((1, input_dim)))
    _test_lipschitz_linear(p, model)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_lipschitz_sequential_non_linear(p):
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    _test_lipschitz_non_linear(p, model)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_lipschitz_sequential_non_linear(p):
    keras.config.set_image_data_format("channels_first")
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    _test_lipschitz_non_linear(p, model, input_dim)
