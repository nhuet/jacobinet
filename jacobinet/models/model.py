import keras
from keras.layers import Layer, Input, InputLayer
from keras.models import Model, Sequential
from .node import get_backward_node
from .base_model import BackwardModel
from jacobinet.layers.layer import BackwardLayer
from .utils import get_gradient
import numpy as np

from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List



def get_backward_model(
    model: Model,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    extra_inputs: Union[List[Input]] = [],
):
    # find output_nodes
    grad_input=[]

    if gradient is not None:
        if not isinstance(gradient, List):
            gradient = [gradient]

        for grad in gradient:
            if hasattr(grad, '_keras_history'):
                grad_input_i = isinstance(grad._keras_history.operation, InputLayer)
            else:
                grad_input_i = False

            grad_input.append(grad_input_i)

    model_outputs = model.output
    model_inputs = model.input
    if not isinstance(model_outputs, list):
        model_outputs = [model_outputs]
    if not isinstance(model_inputs, list):
        model_inputs = [model_inputs]

    """
    output_names = [o.name for o in model_outputs]
    output_nodes = []
    nodes_names = []
    for _, nodes in model._nodes_by_depth.items():
        for node in nodes:
            if node.operation.output.name in output_names:
                output_nodes.append(node)
    """
    
    output_nodes = []
    for model_output in model_outputs:
        node_index = model_output._keras_history.node_index
        nodes = model._nodes_by_depth[node_index]
        node = [node for node in nodes if node.operation.output.name == model_output.name][0]
        output_nodes.append(node)  

    # if multiple output, merge the backward given the different inputs
    
    # if gradient is None: create inputs for backward mask ...
    if gradient is None:
        gradient = [
            Input(list(output_i.shape[1:])) for output_i in model_outputs
        ]
        grad_input = [True]*len(model_outputs)

    if isinstance(gradient, list):
        assert len(gradient) == len(
            output_nodes
        ), "Mismatch between gradient and output nodes: The gradient must be specified for every output node, or not specified at all."
    else:
        gradient = [gradient]

    gradient = [get_gradient(grad, model_inputs) for grad in gradient]

    outputs = []
    is_linear = True

    dico_input={}
    for grad, output_node in zip(gradient, output_nodes):

        output_node, is_linear_node, keep_branch = get_backward_node(
            output_node, grad, mapping_keras2backward_classes
        )
        outputs.append(output_node)
        is_linear = min(is_linear, is_linear_node)

    inputs = []
    # check if the model is linear
    if not is_linear:
        inputs = [inp for inp in model_inputs]
    inputs += extra_inputs
        
    inputs += [gradient[i] for i in range(len(model_outputs)) if grad_input[i]]

    if len(inputs) == 1:
        inputs = inputs[0]
    if len(outputs) == 1:
        outputs = outputs[0]

    return BackwardModel(inputs, outputs)
