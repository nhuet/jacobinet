from keras.layers import Input, Layer, InputLayer
from keras.models import Sequential, Model
from .base_model import BackwardModel, BackwardSequential

from jacobinet import get_backward_layer
from jacobinet.layers.layer import (
    BackwardLayer,
    BackwardLinearLayer,
    BackwardNonLinearLayer,
)
from .utils import get_gradient, is_linear_layer
from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List, Callable


def get_backward_sequential(
    model: Sequential,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    extra_inputs: List[Input] = [],
    get_backward:Callable = get_backward_layer
):
    # get input_dim without batch
    input_dim_wo_batch = list(model.inputs[0].shape[1:])
    # for output_tensor in model.outputs:
    output_dim_wo_batch = list(model.outputs[0].shape[1:])

    grad_input = False
    if gradient is not None:
        if isinstance(gradient, List):
            grad = gradient[0]
        else:
            grad = gradient

        if hasattr(grad, '_keras_history'):
            grad_input = isinstance(grad._keras_history.operation, InputLayer)
        else:
            grad_input = False
                
    # convert every layers
    layers_backward = [
        get_backward(
            layer,
            mapping_keras2backward_classes=mapping_keras2backward_classes,
        )
        for layer in model.layers
    ]
    # check if the layers are all linear
    is_linear = min(
        [
            is_linear_layer(layer_backward)
            for layer_backward in layers_backward
        ]
    )
    if is_linear:
        if gradient is None:
            backward_model = BackwardSequential(layers=layers_backward[::-1])
            # init shape
            backward_model(model.outputs)
            return backward_model
        elif grad_input or len(extra_inputs):
            output = None
            for layer in layers_backward[::-1]:
                if output is None:
                    output = layer(gradient)
                else:
                    output = layer(gradient)
            if grad_input:
                return BackwardModel(output, gradient)
            else:
                return BackwardModel(extra_inputs, output)
        else:
            raise NotImplementedError()

    else:
        input_tensor = Input(input_dim_wo_batch)

        if gradient is None:
            gradient = Input(output_dim_wo_batch)
            grad_input = True
        else:
            gradient = get_gradient(gradient, input_tensor)
        
        # forward propagation
        dico_input_layer = dict()
        output = None
        for layer, backward_layer in zip(model.layers, layers_backward):
            if output is None:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = input_tensor
                elif isinstance(backward_layer, BackwardModel) and not backward_layer.is_linear: 
                    dico_input_layer[id(backward_layer)] = input_tensor
                output = layer(input_tensor)
            else:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = output
                elif isinstance(backward_layer, BackwardModel) and not backward_layer.is_linear: 
                    dico_input_layer[id(backward_layer)] = output
                output = layer(output)

        output_backward = None
        for backward_layer in layers_backward[::-1]:

            if output_backward is None:
                input_backward = [gradient]
            else:
                input_backward = [output_backward]
            if isinstance(backward_layer, BackwardLinearLayer):
                # no need for forward input
                output_backward = backward_layer(input_backward[0])
            elif hasattr(backward_layer, 'is_linear') and backward_layer.is_linear:
                # no need for forward input
                output_backward = backward_layer(input_backward[0])
            elif (isinstance(backward_layer, BackwardModel) or isinstance(backward_layer, BackwardSequential)):
                input_forward = dico_input_layer[id(backward_layer)]
                input_backward = [input_forward]+input_backward
                output_backward = backward_layer(input_backward)
            else:
                input_forward = dico_input_layer[id(backward_layer)]
                #input_backward = [input_forward]+input_backward
                input_backward.append(input_forward)
                output_backward = backward_layer(input_backward)

        inputs = [input_tensor]+extra_inputs
        if grad_input:
            inputs.append(gradient)

        return BackwardModel(inputs, output_backward)
