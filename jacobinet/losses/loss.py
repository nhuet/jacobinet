import numpy as np
import keras
from keras.layers import Layer
import numpy as np
from keras.losses import Loss, CategoricalCrossentropy


CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
# MeanSquaredError
# MeanAbsoluteError
# MeanAbsoluteError
# MeanSquaredLogarithmicError
# CosineSimilarity
# LogCosh
# Hinge
# SquaredHinge
# CategoricalHinge
# KLDivergence
# BinaryCrossentropy

import keras

from typing import Callable, Optional
from keras import KerasTensor as Tensor


def deserialize(name: str) -> Loss:
    """Get the activation from name.

    Args:
        name: name of the method.
    among the implemented Keras activation function.

    Returns:
        the activation function
    """
    name = name.lower()
    if name == CATEGORICAL_CROSSENTROPY:
        return keras.losses.CategoricalCrossentropy()
    raise ValueError("unknown loss")


class Loss_Layer(Layer):
    def __init__(self, loss:Loss, **kwargs):
        super(Loss_Layer, self).__init__(**kwargs)
        self.loss = loss

    def call(self, inputs_):

        y_true, y_pred = inputs_
        # reshape to have batch dimension
        shape_wo_batch = np.prod(y_true.shape[1:])
        y_true_flatten = keras.ops.reshape(y_true, [-1, shape_wo_batch])
        loss = self.loss(y_true, y_pred)

        return keras.ops.zeros_like(y_true_flatten[:,:1]) + loss
    
    # serialization to do
    
class CategoricalCrossentropy_Layer(Loss_Layer):
    loss: CategoricalCrossentropy


default_mapping_loss2layer: dict[type[Loss], type[Loss_Layer]] = {
    # convolution
    CategoricalCrossentropy: CategoricalCrossentropy_Layer,
}

def get_loss_as_layer(loss:Loss, 
                      mapping_loss2layer:Optional[dict[type[Loss], type[Loss_Layer]]]=None,
                      **kwargs):
    
    keras_class = type(loss)

    if mapping_loss2layer is not None:
        default_mapping_loss2layer.update(mapping_loss2layer)

    get_loss_as_layer = default_mapping_loss2layer.get(keras_class)
    if get_loss_as_layer is None:
        raise ValueError(
            "The mapping from the current loss to a layer is not native and not available in mapping_loss2layer, {} not found".format(
                keras_class
            )
        )
    
    return get_loss_as_layer(loss)

