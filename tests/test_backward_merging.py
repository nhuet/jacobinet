from .conftest import compute_output
import keras
from keras.layers import Add
from keras.layers import Dense, Reshape, ReLU, Conv2D, Input
from keras.models import Sequential, Model
from jacobinet.models.sequential import get_backward_sequential
from jacobinet.models.model import get_backward_model
from .conftest import compute_backward_model, serialize_model
import numpy as np
import torch


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
    
    #for i in range(4):
    #    compute_backward_model((input_dim,), model, backward_model, i)
    #
    serialize_model([input_dim, 4], backward_model)
    


def test_backward_add():
    merge_layer = Add()
    n_inputs = 2
    _test_model_multiD_multi_output(merge_layer, n_inputs)
