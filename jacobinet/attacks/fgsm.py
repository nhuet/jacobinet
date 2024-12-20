# from backward
import keras.ops as K
from keras.layers import Layer, Input
from keras.losses import Loss
import numpy as np
from jacobinet.models import BackwardModel, BackwardSequential

from jacobinet.attacks import get_adv_model_base, AdvLayer, AdvModel
from jacobinet.utils import to_list
from .utils import FGSM


import keras
from keras import KerasTensor as Tensor
from typing import Any, Union, List


from typing import Union, List

class FastGradientSign(AdvLayer):

    def __init__(
        self,
        epsilon=0.,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = keras.Variable(epsilon, trainable=False)

    # @keras.ops.custom_gradient
    def call(self, inputs, training=None, mask=None):
        # inputs = [x and \delta f(x)]
        x, grad_x = inputs

        def grad(*args, upstream=None):
            import pdb

            pdb.set_trace()
            return keras.ops.tanh(upstream)

        # project given lp norm
        adv_x = x + self.epsilon * keras.ops.sign(-grad_x)
        adv_x = self.project_lp_ball(adv_x)
        return keras.ops.clip(adv_x, self.lower, self.upper)


    # saving
    def get_config(self):
        config = super().get_config()
        eps_config = keras.saving.serialize_keras_object(self.epsilon)
        config["epsilon"] = eps_config
        config["lower"] = self.lower
        config["upper"] = self.upper
        return config
    

def get_fgsm_model(model, 
                  loss:Union[str, Loss, Layer]='categorical_crossentropy',
                  mapping_keras2backward_classes={}, # define type
                  mapping_keras2backward_losses={},
                  **kwargs)->AdvModel: # we do not compute gradient on extra_inputs, loss should return (None, 1)
    
    base_adv_model:BackwardModel = get_adv_model_base(
                                        model=model, 
                                        loss=loss, 
                                        mapping_keras2backward_classes=mapping_keras2backward_classes,
                                        mapping_keras2backward_losses=mapping_keras2backward_losses,
                                        **kwargs)
    
    inputs:List[Tensor] = to_list(base_adv_model.inputs)
    adv_pred:List[Tensor] = to_list(base_adv_model.outputs)
    lower = -np.inf
    upper = np.inf
    p=-1
    radius = np.inf
    if 'lower' in kwargs:
        lower = kwargs['lower']
    if 'upper' in kwargs:
        upper = kwargs['upper']
    if 'p' in kwargs:
        p = kwargs['p']
    if 'radius' in kwargs:
        radius = kwargs['radius']

    # use lp norm as well
    if 'epsilon' in kwargs:
        fgsm_layer = FastGradientSign(epsilon=kwargs['epsilon'], lower=lower, upper=upper, p=p, radius=radius)
    else:
        fgsm_layer = FastGradientSign(epsilon=kwargs['epsilon'], lower=lower, upper=upper, p=p, radius=radius)

    output = fgsm_layer(inputs[:-1]+adv_pred)

    fgsm_model = AdvModel(inputs=inputs,
                          outputs=output,
                          layer_adv= fgsm_layer,
                          backward_model=base_adv_model, 
                          method=FGSM
    )
    
    return fgsm_model



