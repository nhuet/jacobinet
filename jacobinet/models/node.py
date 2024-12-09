from keras import KerasTensor as Tensor
from keras.layers import Layer, InputLayer
from keras.models import Model, Sequential
from keras.src.ops.node import Node
import keras.ops as K
from typing import List, Union, Optional, Callable
from jacobinet.layers.layer import BackwardLayer, BackwardLinearLayer
from jacobinet.models.utils import get_backward, FuseGradients

import numpy as np



def get_backward_node(
    node: None,
    gradient: Tensor,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    input_name=None
):
    # step 1: get parents
    parent_nodes: List[None] = node.parent_nodes

    # step 2: get input of node - warning map i with the right parent
    input_node: Union[Tensor, List[Tensor]]

    # step 3: get layer from node
    layer_node: Union[Layer, Model, Sequential] = node.operation

    # step 3.1 if layer is an InputLayer stop the algorithm
    if isinstance(layer_node, InputLayer):
        # check we reach the right input
        if input_name is None:
            return gradient, True, True
        else:
            return gradient, True, layer_node.output.name==input_name

    # step 4: get backward layer
    backward_layer_node: BackwardLayer = get_backward(
        layer_node,
        gradient_shape=list(gradient.shape[1:]),
        mapping_keras2backward_classes=mapping_keras2backward_classes,
    )
    is_linear = True
    if isinstance(backward_layer_node, BackwardLinearLayer):
        gradients = backward_layer_node(gradient)
    else:
        layer_node_inputs = layer_node.input
        if not isinstance(layer_node_inputs, list):
            layer_node_inputs = [layer_node_inputs]
        gradients = backward_layer_node([gradient] + layer_node_inputs)
        is_linear = False

    keep_output = True
    if backward_layer_node.n_input != 1:
        results = [
            get_backward_node(p_node, p_grad, mapping_keras2backward_classes, input_name)
            for (p_node, p_grad) in zip(parent_nodes, gradients)
        ]

        results = [r for r in results if r[-1]] # keep the parents whom is connected to the right input

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
            output= None
            is_linear = False
            keep_output = False
    else:
        result = get_backward_node(parent_nodes[0], gradients, mapping_keras2backward_classes, input_name)
        output = result[0]
        keep_output = result[-1]
        if is_linear:
            is_linear = min(is_linear, result[1])

    return output, is_linear, keep_output
