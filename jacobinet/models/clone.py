from keras.models import Model, Sequential
from keras.layers import Layer, Input
from jacobinet.layers.layer import BackwardLayer
from jacobinet import get_backward_layer
from jacobinet.models.model import get_backward_functional
from jacobinet.models.sequential import get_backward_sequential
from jacobinet.models.utils import to_list

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
    def get_backward(layer, mapping_keras2backward_classes, **kwargs):
        if isinstance(layer, Sequential):
            if not input_mask is None:
                 input_mask_ = to_list(input_mask)
                 if layer.input.name in input_mask_:
                    raise NotImplementedError()
            return get_backward_sequential(model= layer,mapping_keras2backward_classes = mapping_keras2backward_classes, get_backward=get_backward)
        elif isinstance(layer, Model):

            return get_backward_functional(model= layer,
                                            mapping_keras2backward_classes = mapping_keras2backward_classes, 
                                            input_mask=input_mask, get_backward=get_backward,
                                            )
        elif isinstance(layer, Layer):
            return get_backward_layer(layer, mapping_keras2backward_classes)
        else: 
            raise ValueError('unsupported object layer={}'.format(layer))
        
    if isinstance(model, Sequential):
        if not input_mask is None:
                 raise NotImplementedError()
        backward_model= get_backward_sequential(
            model, gradient, mapping_keras2backward_classes, extra_inputs, get_backward=get_backward
        )
        #backward_model.set_model(model)
    else:
        backward_model= get_backward_functional(
            model, gradient, mapping_keras2backward_classes, extra_inputs, input_mask, get_backward=get_backward
        )
    return backward_model
    
