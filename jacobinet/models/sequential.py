from keras.layers import Input, Layer
from keras.models import Sequential, Model
from .base_model import BackwardModel

from jacobinet import get_backward_layer
from jacobinet.layers.layer import (
    BackwardLayer,
    BackwardLinearLayer,
    BackwardNonLinearLayer,
)
from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List


def get_backward_sequential(
    model: Sequential,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    use_gradient_as_backward_input: bool = False,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    extra_inputs: Union[List[Input], None] = None,
):
    # to do implement gradient

    # convert every layers
    layers_backward = [
        get_backward_layer(
            layer,
            mapping_keras2backward_classes=mapping_keras2backward_classes,
        )
        for layer in model.layers
    ]
    # check if the layers are all linear
    is_linear = min(
        [
            isinstance(layer_backward, BackwardLinearLayer)
            for layer_backward in layers_backward
        ]
    )
    if is_linear:
        backward_model = Sequential(layers=layers_backward[::-1])
        # init shape
        backward_model(model.outputs)
        return backward_model
    else:
        # get input_dim without batch
        input_dim_wo_batch = list(model.inputs[0].shape[1:])
        # for output_tensor in model.outputs:
        output_dim_wo_batch = list(model.outputs[0].shape[1:])

        backward_input_tensor = Input(output_dim_wo_batch)
        input_tensor = Input(input_dim_wo_batch)
        # forward propagation
        dico_input_layer = dict()
        output = None
        for layer, backward_layer in zip(model.layers, layers_backward):
            if output is None:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = input_tensor
                output = layer(input_tensor)
            else:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = output
                output = layer(output)

        gradient = None
        for backward_layer in layers_backward[::-1]:

            if isinstance(backward_layer, BackwardLinearLayer):
                # no need for forward input
                if gradient is None:
                    gradient = backward_layer(backward_input_tensor)
                else:
                    gradient = backward_layer(gradient)
            else:
                input_forward = dico_input_layer[id(backward_layer)]
                if gradient is None:
                    gradient = backward_layer(
                        [backward_input_tensor, input_forward]
                    )
                else:
                    gradient = backward_layer([gradient, input_forward])

        return Model([input_tensor, backward_input_tensor], gradient)
