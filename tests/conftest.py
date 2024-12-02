import os
import numpy as np
from numpy.testing import assert_almost_equal
import keras
from keras.models import Sequential
import torch

from jacobinet.models import is_linear


def is_invertible(layer, backward_layer):

    input_shape = list(layer.input.shape[1:])
    batch_size = 10
    input_random = np.reshape(
        np.random.rand(np.prod(input_shape) * batch_size),
        [batch_size] + input_shape,
    )
    model = Sequential([layer, backward_layer])
    output_random = model.predict(input_random)
    assert_almost_equal(input_random, output_random)


def linear_mapping(layer, backward_layer):

    input_shape = list(layer.input.shape[1:])
    output_shape = list(layer.output.shape[1:])
    n_input = np.prod(input_shape)
    n_output = np.prod(output_shape)
    weights_in = np.reshape(np.eye(n_input), [n_input] + input_shape)

    model_layer = Sequential([layer])
    model_backward = Sequential([backward_layer])
    w_f = model_layer.predict(weights_in, verbose=0)
    w_f = np.reshape(w_f, [n_input, n_output])

    weights_out = np.reshape(np.eye(n_output), [n_output] + output_shape)
    w_b = model_backward.predict(weights_out, verbose=0)
    w_b = np.reshape(w_b, [n_output, n_input])
    w_b = w_b.T
    assert_almost_equal(w_f, w_b, decimal=3)


def serialize(layer, backward_layer):

    input_shape = list(layer.input.shape[1:])
    batch_size = 10
    input_random = np.reshape(
        np.random.rand(np.prod(input_shape) * batch_size),
        [batch_size] + input_shape,
    )
    toy_model = Sequential([layer, backward_layer])

    filename = "test_serialize_{}_{}.keras".format(
        layer.__class__.__name__, layer.name
    )
    # detach toy model to cpu
    # toy_model.to('cpu')
    toy_model.save(filename)  # The file needs to end with the .keras extension
    output_before_export = toy_model(input_random).cpu().detach().numpy()

    # deserialize
    load_model_ = keras.models.load_model(filename)

    # compare with the previous output
    output_after_export = load_model_(input_random).cpu().detach().numpy()
    os.remove(filename)
    np.testing.assert_almost_equal(
        output_before_export, output_after_export, err_msg="corrupted weights"
    )


def serialize_model(list_input_dim, backward_model):

    batch_size = 10
    inputs = []
    for input_dim in list_input_dim:
        input_random = np.reshape(
            np.random.rand(input_dim * batch_size), [batch_size, input_dim]
        )
        inputs.append(input_random)

    if len(inputs) == 1:
        inputs = inputs[0]

    filename = "test_serialize_{}_{}.keras".format(
        backward_model.__class__.__name__, backward_model.name
    )
    # detach toy model to cpu
    # toy_model.to('cpu')
    output_before_export = backward_model.predict(inputs)
    backward_model.save(
        filename
    )  # The file needs to end with the .keras extension

    # deserialize
    load_model = keras.models.load_model(filename)

    # compare with the previous output

    output_after_export = load_model.predict(inputs)
    os.remove(filename)
    np.testing.assert_almost_equal(
        output_before_export, output_after_export, err_msg="corrupted weights"
    )


def compute_backward_layer(
    input_shape, model, backward_model, input_random=False
):
    if input_random:
        input_ = torch.randn(1, input_shape[0], requires_grad=True)
    else:
        input_ = torch.ones((1, input_shape[0]), requires_grad=True)

    output = model(input_)
    select_output = output[0, 0]
    select_output.backward()
    gradient = input_.grad.cpu().detach().numpy()

    mask_output = torch.Tensor([1] + [0] * 31)[None]

    gradient_ = backward_model([mask_output, input_]).cpu().detach().numpy()
    assert_almost_equal(gradient, gradient_)


def compute_backward_model(
    input_shape, model, backward_model, index_output=0, grad_value=None
):
    input_ = torch.randn(np.prod(input_shape), requires_grad=True)
    input_reshape = torch.reshape(input_, [1] + list(input_shape))
    output = model(input_reshape)
    select_output = output[0, index_output]
    select_output.backward()
    gradient = input_.grad.cpu().detach().numpy()

    if grad_value is None:
        gradient_value = np.eye(output.shape[-1])
        mask_output = torch.Tensor(gradient_value[index_output])[None]
    else:
        mask_output = grad_value

    if is_linear(backward_model):
        gradient_ = backward_model(mask_output).cpu().detach().numpy()
    else:
        if (
            isinstance(backward_model.input, list)
            and len(backward_model.input) == 2
        ):
            gradient_ = (
                backward_model([input_reshape, mask_output])
                .cpu()
                .detach()
                .numpy()
            )
        else:
            gradient_ = backward_model(input_reshape).cpu().detach().numpy()
    gradient = np.reshape(gradient, input_shape)
    assert_almost_equal(gradient, gradient_[0])


def compute_output(input_, layers):
    output = None
    for layer in layers:
        if output is None:
            output = layer(input_)
        else:
            output = layer(output)
    return output
