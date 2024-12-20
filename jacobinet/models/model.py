import keras
from keras.layers import Layer, Input, InputLayer
from keras.models import Model, Sequential
from .node import get_backward_node
from .base_model import BackwardModel
from jacobinet.layers.layer import BackwardLayer
from .utils import get_gradient, FuseGradients, to_list
from jacobinet import get_backward_layer


import numpy as np

from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List


def get_backward_functional(
    model: Model,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    extra_inputs: Union[List[Input]] = [],
    input_mask=None,
    get_backward: callable = get_backward_layer,
):
    # find output_nodes
    grad_input = (
        []
    )  # list of boolean to check whether the gradient should be used as an input of the backward model
    if gradient is not None:
        if not isinstance(gradient, List):
            gradient = [gradient]

        for grad in gradient:
            if grad is None:
                raise ValueError(
                    "None values among not None values of gradient is not considered. Mismatch between gradient and output nodes: The gradient must be specified for every output node, or not specified at all."
                )
            if hasattr(grad, "_keras_history"):
                grad_input_i = isinstance(
                    grad._keras_history.operation, InputLayer
                )
            else:
                grad_input_i = False

            grad_input.append(grad_input_i)

    # convert model inputs and outputs as list
    model_inputs = to_list(model.input)
    model_outputs = to_list(model.output)

    if input_mask is None:
        # keep every input
        # input_mask=[input_.name for input_ in model_inputs]
        input_mask = []

    output_nodes = []
    for model_output in model_outputs:
        node_index = model_output._keras_history.node_index
        nodes = model._nodes_by_depth[node_index]
        node = [
            node
            for node in nodes
            if node.operation.output.name == model_output.name
        ][0]
        output_nodes.append(node)

    # if gradient is None: create inputs for backward mask
    if gradient is None:
        gradient = [
            Input(
                list(output_i.shape[1:]),
                name="{}_{}".format(output_i.name, "gradient"),
            )
            for output_i in model_outputs
        ]
        grad_input = [True] * len(model_outputs)

    if isinstance(gradient, list):
        assert len(gradient) == len(
            output_nodes
        ), "Mismatch between gradient and output nodes: The gradient must be specified for every output node, or not specified at all."
    else:
        gradient = [gradient]

    gradient = [get_gradient(grad, model_inputs) for grad in gradient]

    outputs = []
    is_linear = True

    for input_ in model_inputs:
        input_name = input_.name
        if not input_name in input_mask:

            outputs_i = []
            is_linear_i = True

            for i, (grad, output_node) in enumerate(
                zip(gradient, output_nodes)
            ):
                output_node, is_linear_node, keep_branch = get_backward_node(
                    output_node,
                    grad,
                    mapping_keras2backward_classes,
                    input_name=input_name,
                    get_backward=get_backward,
                )
                if keep_branch:
                    outputs_i.append(output_node)
                    is_linear_i = min(is_linear_i, is_linear_node)

            if not len(outputs_i):
                continue
            else:
                outputs += outputs_i
                is_linear = min(is_linear, is_linear_i)

    inputs = []
    # check if the model is linear
    if not is_linear:
        inputs = [
            inp for inp in model_inputs
        ]  # due to masking some inputs may be not required to produce the input (aka linear branches of the model). Up to now we have no way to track it
    inputs += extra_inputs

    inputs += [gradient[i] for i in range(len(model_outputs)) if grad_input[i]]

    if len(inputs) == 1:
        inputs = inputs[0]
    if len(outputs) == 1:
        outputs = outputs[0]

    # or dictionary
    return BackwardModel(inputs, outputs, n_input=len(model.inputs))
