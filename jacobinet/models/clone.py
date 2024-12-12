from keras.models import Model, Sequential
from keras.layers import Layer, Input
from .sequential import get_backward_sequential
from .model import get_backward_functional

from jacobinet.layers.layer import BackwardLayer

from keras import KerasTensor as Tensor
from typing import Union, Optional, Tuple, Any, List
from typing import Optional


def clone_to_backward(
    model: Model,
    gradient: Union[None, Tensor, List[Tensor]] = None,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    extra_inputs: Union[List[Input]] = [],
    input_mask=None,
):
    if isinstance(model, Sequential):
        return get_backward_sequential(
            model, gradient, mapping_keras2backward_classes, extra_inputs
        )
    else:
        return get_backward_functional(
            model, gradient, mapping_keras2backward_classes, extra_inputs, input_mask
        )
    
