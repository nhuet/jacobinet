import keras
from keras.layers import Layer, Input, InputLayer
from keras.models import Model, Sequential
from .node import get_backward_node
from jacobinet.layers.layer import BackwardLayer
from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List


class GradConstant(Layer):

    def __init__(self, gradient, **kwargs):
        """
        to fill
        """
        super(GradConstant, self).__init__(**kwargs)
        self.grad_const = keras.ops.convert_to_tensor(gradient)

    def call(self, inputs_):
        return self.grad_const

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grad_const": keras.saving.serialize_keras_object(
                    self.grad_const
                ),
            }
        )
        return config

    def compute_output_shape(self, input_shape):

        return list(self.grad_const.shape)

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("grad_const")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)


def get_gradient(grad: Tensor, input) -> Tuple[Any, bool]:
    # if grad is an InputLayer return grad
    if isinstance(grad, InputLayer) or isinstance(grad, keras.KerasTensor):
        return grad  # grad is a KerasTensor that come from input or extra_inputs or is an Input Tensor
    # else return it as a Constant of a layer
    constant = GradConstant(gradient=grad)(input)
    return constant


def get_backward_model(
    model: Model,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    extra_inputs: Union[List[Input]] = [],
):
    # find output_nodes
    grad_input = isinstance(gradient, InputLayer)
    model_outputs = model.output
    model_inputs = model.input
    if not isinstance(model_outputs, list):
        model_outputs = [model_outputs]
    if not isinstance(model_inputs, list):
        model_inputs = [model_inputs]
    output_names = [o.name for o in model_outputs]
    output_nodes = []
    nodes_names = []
    for _, nodes in model._nodes_by_depth.items():
        for node in nodes:
            if node.operation.output.name in output_names:
                output_nodes.append(node)

    # if gradient is None: create inputs for backward mask ...
    if gradient is None:
        gradient = [
            Input(list(output_i.shape[1:])) for output_i in model_outputs
        ]
        use_gradient_as_backward_input = True
        grad_input = True

    if isinstance(gradient, list):
        assert len(gradient) == len(
            output_nodes
        ), "Mismatch between gradient and output nodes: The gradient must be specified for every output node, or not specified at all."
    else:
        gradient = [gradient]

    gradient = [get_gradient(grad, model_inputs) for grad in gradient]

    outputs = []
    is_linear = True
    for grad, output_node in zip(gradient, output_nodes):

        output_node, is_linear_node = get_backward_node(
            output_node, grad, mapping_keras2backward_classes
        )
        outputs.append(output_node)
        is_linear = min(is_linear, is_linear_node)

    inputs = []
    # check if the model is linear
    if not is_linear:
        inputs = [inp for inp in model_inputs]
    inputs += extra_inputs
    if grad_input:
        inputs += gradient

    if len(inputs) == 1:
        inputs = inputs[0]
    if len(outputs) == 1:
        outputs = outputs[0]

    return Model(inputs, outputs)
