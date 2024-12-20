from keras import KerasTensor as Tensor
from keras.layers import Layer, InputLayer, Input
from keras.models import Model, Sequential
from .base_model import BackwardModel, BackwardSequential
from keras.src.ops.node import Node
import keras.ops as K
from typing import List, Union, Optional, Callable
from jacobinet.layers.layer import BackwardLayer, BackwardLinearLayer
from jacobinet.models.utils import get_backward, FuseGradients, to_list
from jacobinet import get_backward_layer

import numpy as np


def get_backward_node(
    node: None,
    gradient: Tensor,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    input_name=None,
    get_backward: Callable = get_backward_layer,
):
    # step 1: get parents
    parent_nodes: List[None] = node.parent_nodes

    # step 2: get layer from node
    layer_node: Union[Layer, Model, Sequential] = node.operation

    # step 3.1 if layer is an InputLayer stop the algorithm
    if isinstance(layer_node, InputLayer):
        # check we reach the right input
        if input_name is None:
            return gradient, True, True
        else:
            return gradient, True, layer_node.output.name == input_name

    # step 4: get backward layer
    backward_layer_node: BackwardLayer = get_backward(
        layer_node,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
    )

    is_linear = True
    if isinstance(backward_layer_node, BackwardLinearLayer):
        gradients = backward_layer_node(gradient)
    elif isinstance(layer_node, Sequential):
        if backward_layer_node.is_linear:
            gradients = backward_layer_node(gradient)
        else:
            is_linear = False
            layer_node_inputs = to_list(layer_node.input)
            gradients = backward_layer_node(layer_node_inputs + [gradient])
    elif isinstance(layer_node, Model):
        if backward_layer_node.is_linear:
            gradients = backward_layer_node(gradient)
        else:
            is_linear = False
            # warning: in case of nested models the following can happen: node.operation.input != node.parent_nodes[0].operation.output
            # consider working on the parents'outputs to avoid disconnected graph
            layer_node_inputs = []
            for parent_node in parent_nodes:
                layer_node_inputs = to_list(parent_node.operation.output)
            gradients = backward_layer_node(layer_node_inputs + [gradient])
    else:
        layer_node_inputs = to_list(layer_node.input)
        gradients = backward_layer_node([gradient] + layer_node_inputs)
        is_linear = False

    keep_output = True
    if backward_layer_node.n_input != 1:
        results = [
            get_backward_node(
                p_node,
                p_grad,
                mapping_keras2backward_classes,
                input_name,
                get_backward=get_backward,
            )
            for (p_node, p_grad) in zip(parent_nodes, gradients)
        ]

        results = [
            r for r in results if r[-1]
        ]  # keep the parents whom is connected to the right input

        if len(results):
            outputs = [
                r[0] for r in results
            ]  # output tensor is the first output of get_backward_node

            if is_linear:
                is_linear = min(is_linear, min([r[1] for r in results]))
            # chain rule along multiple inputs: expand, concat and sum
            fuse_layer = FuseGradients()
            output = fuse_layer(outputs)
        else:
            output = None
            is_linear = False
            keep_output = False
    else:
        result = get_backward_node(
            parent_nodes[0],
            gradients,
            mapping_keras2backward_classes,
            input_name,
            get_backward=get_backward,
        )
        output = result[0]
        keep_output = result[-1]
        if is_linear:
            is_linear = min(is_linear, result[1])

    return output, is_linear, keep_output
