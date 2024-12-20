# from backward
import keras.ops as K
from keras.layers import Layer, Input
from keras.losses import Loss
from keras.layers import RNN
import numpy as np
from jacobinet.models import BackwardModel

from jacobinet.attacks import get_adv_model_base, AdvLayer, AdvModel
from jacobinet.attacks.fgsm import get_fgsm_model
from jacobinet.utils import to_list
from .utils import PGD, FGSM


import keras
from keras import KerasTensor as Tensor
from typing import Any, Union, List


from typing import Union, List

# First, let's define a RNN Cell, as a layer subclass.
class PGD_Cell(keras.Layer):

    def __init__(self, fgsm_model, **kwargs):
        super().__init__(**kwargs)
        self.fgsm_model = fgsm_model
        self.state_size = 1

    def call(self, y, states):
        x = states[0] # x
        # get adversarial attack using fgsm_model
        adv_x = self.fgsm_model([x, y])
        return adv_x, [adv_x]


class ProjectedGradientDescent(AdvLayer):

    def __init__(
        self,
        n_iter:int=10,
        fgsm_model:AdvModel=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.fgsm_layer = fgsm_model.layer_adv
        pgd_cell = PGD_Cell(fgsm_model=fgsm_model)
        self.inner_layer = RNN(pgd_cell)

        # init attributes with fgsm_layer
        self.upper = self.fgsm_layer.upper
        self.lower = self.fgsm_layer.lower
        self.p = self.fgsm_layer.p
        self.radius = self.fgsm_layer.radius

    # saving
    def get_config(self):
        config = super().get_config()
        inner_layer_config = keras.saving.serialize_keras_object(self.inner_layer)
        config['gpd_cell'] = inner_layer_config
        config['n_iter'] = self.n_iter

        return config
    
    def set_upper(self, upper):
        self.fgsm_layer.set_upper(upper)
        self.upper = self.fgsm_layer.upper

    def set_lower(self, lower):
        self.fgsm_layer.set_lower(lower)
        self.lower = self.fgsm_layer.lower

    def set_p(self, p):
        self.fgsm_layer.set_p(p)
        self.p = self.fgsm_layer.p

    def set_radius(self, radius):
        self.fgsm_layer.set_radius(radius)
        self.radius = self.fgsm_layer.radius

    def call(self, inputs, training=None, mask=None):

        x, y = inputs
        z = keras.ops.repeat(keras.ops.expand_dims(y, 1), self.n_iter, 1)

        return self.inner_layer(z, initial_state=[x])


def get_pgd_model(model, 
                  loss:Union[str, Loss, Layer]='categorical_crossentropy',
                  mapping_keras2backward_classes={}, # define type
                  mapping_keras2backward_losses={},
                  **kwargs)->AdvModel: # we do not compute gradient on extra_inputs, loss should return (None, 1)
    
    fgsm_model = get_fgsm_model(model, 
                                loss=loss, 
                                mapping_keras2backward_classes=mapping_keras2backward_classes, # define type
                                mapping_keras2backward_losses=mapping_keras2backward_losses,
                                **kwargs)
    n_iter = 10
    if 'n_iter' in kwargs:
        n_iter = kwargs['n_iter']
    pgd_layer = ProjectedGradientDescent(n_iter=n_iter, fgsm_model=fgsm_model)
    inputs:List[Tensor] = to_list(fgsm_model.inputs)

    output = pgd_layer(inputs)

    pgd_model = AdvModel(inputs=inputs,
                          outputs=output,
                          layer_adv= pgd_layer,
                          backward_model=fgsm_model.backward_model, 
                          method=PGD
    )
    
    return pgd_model