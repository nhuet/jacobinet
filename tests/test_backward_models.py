import keras
from keras.layers import (
    Dense,
    Reshape,
    Flatten,
    ReLU,
    Conv2D,
    DepthwiseConv2D,
    Input,
)
from keras.models import Sequential, Model
from jacobinet.models.sequential import get_backward_sequential
from jacobinet.models.model import get_backward_functional
from .conftest import compute_backward_model, serialize_model, compute_output
import numpy as np
import torch

# preliminary tests: gradient is derived automatically by considering single output model
"""
def test_sequential_linear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is linear

    _ = backward_model(np.ones((1,1)))
    
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([32], model)
    serialize_model([1], backward_model)


def test_sequential_nonlinear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)

def test_sequential_multiD():

    input_dim = 36
    layers = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(1)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)

def _test_sequential_multiD_channel_last():

    input_dim = 72
    layers = [Reshape((6, 6, 2)), DepthwiseConv2D(2, (3, 3), data_format="channels_last"), ReLU(), Reshape((-1,)), Dense(1)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)

# same using model instead of Sequential
def test_model_linear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    input_ = Input((input_dim,))
    output=None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model)
    # model is linear
    _ = backward_model(np.ones((1,1)))
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([1], backward_model)

def test_model_nonlinear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    input_ = Input((input_dim,))
    output=None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model)
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)

def test_model_multiD():

    input_dim = 36
    layers = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(1)]
    input_ = Input((input_dim,))
    output=None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)

def _test_model_multiD_channel_last():

    input_dim = 72
    layers = [Reshape((6, 6, 2)), Conv2D(2, (3, 3), data_format="channels_last"), ReLU(), Reshape((-1,)), Dense(1)]
    input_ = Input((input_dim,))
    output=None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim, 1], backward_model)


###### encode gradient as a KerasVariable #####
def test_model_multiD_with_gradient_set():

    input_dim = 36
    layers = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(1)]
    gradient = keras.Variable(np.ones((1,1)))
    input_ = Input((input_dim,))
    output=None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model, gradient=gradient)
    # model is not linear
    _ = backward_model(torch.ones((1, input_dim)))
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([input_dim], backward_model)


# extra inputs
def test_model_multiD_extra_input():

    input_dim = 36
    layers = [
        Reshape((1, 6, 6)),
        Conv2D(2, (3, 3)),
        ReLU(),
        Reshape((-1,)),
        Dense(1),
    ]
    input_ = Input((input_dim,))
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    # gradient is the result of extra_inputs
    extra_input = Input((10,))
    gradient = keras.ops.max(extra_input, axis=-1)
    backward_model = get_backward_functional(
        model, gradient=gradient, extra_inputs=[extra_input]
    )
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1, 10))])

    mask_output = torch.eye(10)
    for i in range(10):
        compute_backward_model(
            (input_dim,),
            model,
            backward_model,
            0,
            grad_value=mask_output[i][None],
        )

    serialize_model([input_dim, 10], backward_model)

# multiple outputs
### multi output neural network #####
def test_model_multiD_multi_output():

    input_dim = 36
    layers = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(10)]
    input_ = Input((input_dim,))
    output=None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    model = Model(input_, output)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model, gradient=Input((10,)))
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,10))])

    for i in range(10):
        compute_backward_model((input_dim,), model, backward_model, i)
    
    serialize_model([input_dim, 10], backward_model)#
"""

### multi output neural network #####
def _test_model_multiD_multi_outputs():

    input_dim = 36
    layers_0 = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(10)]
    layers_1 = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(20)]
    input_ = Input((input_dim,))
    output=None
    output_0 = compute_output(input_, layers_0)
    output_1 = compute_output(input_, layers_1)

    model = Model(input_, [output_0, output_1])
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_functional(model, gradient=[Input((10,)), Input((20,))])
    # model is not linear

    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,10)),  torch.ones((1,20))])

    # freeze one model and computer backward on the other branch

    model_0 = Model(input_, output_0)
    backward_model_0 = get_backward_functional(model, gradient=[Input((10,)), keras.Variable(np.zeros((1, 20)))])
    backward_model_0_bis = get_backward_functional(model_0)
    grad_0 = backward_model_0([torch.ones((1, input_dim)), torch.ones((1,10))])
    grad_0_bis = backward_model_0_bis([torch.ones((1, input_dim)), torch.ones((1,10))])
    #import pdb; pdb.set_trace()

    for i in range(10):
        compute_backward_model((input_dim,), model_0, backward_model_0_bis, i)
    """

    mask_output = torch.eye(10)
    for i in range(10):
        compute_backward_model((input_dim,), model, backward_model, i)
    """
    #serialize_model([input_dim, 10], backward_model)


# nested models 
def test_nested_sequential_linear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(input_dim, use_bias=False)]
    inner_model = Sequential(layers)
    _ = inner_model(torch.ones((1, input_dim)))

    layers = [Dense(2, use_bias=False), Dense(input_dim, use_bias=False)]
    inner_model_bis = Sequential(layers)
    _ = inner_model_bis(torch.ones((1, input_dim)))

    model = Sequential([inner_model, inner_model_bis, Dense(1)])
    backward_model = get_backward_sequential(model)
    # model is linear

    _ = backward_model(np.ones((1,1)))
    
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([32], model)
    serialize_model([1], backward_model)


def test_nested_sequential_nonlinear_linear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    nested_model = Sequential(layers)
    _ = nested_model(torch.ones((1, input_dim)))

    backward_model = get_backward_sequential(nested_model)
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1,1))])
    compute_backward_model((input_dim,), nested_model, backward_model)
    serialize_model([input_dim, 1], backward_model)
