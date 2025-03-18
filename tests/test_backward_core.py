import numpy as np
import pytest
from jacobinet import clone_to_backward
from jacobinet import get_backward_layer as get_backward
from keras.layers import Activation, Dense, EinsumDense
from keras.models import Sequential

from .conftest import compute_backward_layer, linear_mapping, serialize


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_backward_Dense(activation):
    layer = Dense(units=3, activation=activation)
    model_layer = Sequential([layer, Dense(1)])
    model_layer_split = Sequential([Dense(units=3), Activation(activation), Dense(1)])
    input_dim = 2
    _ = model_layer(np.ones((input_dim,))[None])
    _ = model_layer_split(np.ones((input_dim,))[None])

    model_layer_split.set_weights(model_layer.get_weights())

    batch = 32
    input = np.reshape(100 * (np.random.rand(batch * input_dim) - 0.5), (batch, input_dim))

    backward_model = clone_to_backward(model_layer)
    backward_model_split = clone_to_backward(model_layer_split)

    output_model = backward_model.predict([input, np.ones((batch, 1))])
    output_model_split = backward_model_split.predict([input, np.ones((batch, 1))])

    np.testing.assert_almost_equal(output_model, output_model_split, err_msg="corrupted weights")


def test_backward_Dense_wo_activation():
    activation = "linear"
    layer = Dense(units=3, use_bias=False, activation=activation)
    model_layer = Sequential([layer])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = Dense(units=3, use_bias=False, activation=activation)
    model_layer = Sequential([layer])
    input_shape = (2, 32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)

    layer_0 = Dense(units=5, use_bias=False, activation=activation)
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
