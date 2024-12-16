# from backward
import keras.ops as K
from keras.layers import Layer
import numpy as np
from jacobinet.models import BackwardModel, BackwardSequential

import keras
from typing import Any, Union, List


from typing import Union, List

class Lipschitz(Layer):

    def __init__(
        self,
        p: float,
        axis:Union[int, list[int]]=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.axis=axis
        if not p in [1, 2, np.inf]:
            raise ValueError('unknown norm for lipschitz bound')

    def call(self, inputs, training=None, mask=None):
        if self.p==np.inf:
            return K.max(K.abs(inputs), axis=self.axis)
        if self.p==2:
            return K.sqrt(K.sum(inputs**2, axis=self.axis))
        if self.p ==1:
            return K.sum(K.abs(inputs), axis=self.axis)



class LipschitzModel(keras.Model):
    def __init__(
        self,
        lipschitz_norm:float,
        backward_model:Union[BackwardModel, BackwardSequential],
        *args,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.lipschitz_norm = lipschitz_norm
        self.backward_model = backward_model

    def get_config(self):
        config = super().get_config()
        lipschitz_config = keras.saving.serialize_keras_object(self.lipschitz_norm)
        config['lipschitz_norm'] = lipschitz_config
        backward_config = keras.saving.serialize_keras_object(self.backward_model)
        config['backward_model'] = backward_config
        return config
