# from backward
from typing import Any, Union

import keras  # type:ignore
import keras.ops as K  # type:ignore
import numpy as np
from jacobinet.models import BackwardModel, BackwardSequential
from keras.layers import Layer  # type:ignore


@keras.saving.register_keras_serializable()
class Lipschitz(Layer):
    """
    A custom Keras layer that computes the Lipschitz bound of the input tensor
    using a specified norm (1, 2, or infinity).

    The Lipschitz bound is a measure of the rate of change of a function with respect
    to its input. This class calculates the Lipschitz norm along a specified axis using
    one of the following norms:
        - p = 1: L1 norm (sum of absolute values)
        - p = 2: L2 norm (Euclidean norm)
        - p = np.inf: L∞ norm (max absolute value)

    Args:
        p : The norm to use for the Lipschitz bound. Must be one of [1, 2, np.inf].
        axis: The axis (or axes) along which to compute the norm.
            Default is -1 (last axis).

    Raises:
        ValueError: If `p` is not one of the valid norms [1, 2, np.inf].

    Example:
        ```python
        lip_layer = Lipschitz(p=2, axis=-1)
        output = lip_layer(inputs)
        ```
    """

    def __init__(
        self,
        p: float,
        axis: Union[int, list[int]] = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.axis = axis
        if not p in [1, 2, np.inf]:
            raise ValueError("unknown norm for lipschitz bound")

    def call(self, inputs, training=None, mask=None):
        if self.p == np.inf:
            return K.max(K.abs(inputs), axis=self.axis)
        if self.p == 2:
            return K.sqrt(K.sum(inputs**2, axis=self.axis))
        if self.p == 1:
            return K.sum(K.abs(inputs), axis=self.axis)


class LipschitzModel(keras.Model):
    """
    A custom Keras model that combines a Lipschitz norm computation with a backward model
    to enable the estimation of the Lipschitz constant of a given function.

    This model is designed to calculate the Lipschitz constant for the function described by
    the `backward_model`. The `lipschitz_norm` is used to specify the type of norm (1, 2, or np.inf)
    that should be computed for the Lipschitz constant.

    Args:
        lipschitz_norm: The norm used to compute the Lipschitz constant.
                                 Should be one of [1, 2, np.inf].
        backward_model: The model or sequence of operations to compute the backward pass.

    Example:
        ```python
        lipschitz_model = LipschitzModel(lipschitz_norm=2, backward_model=my_backward_model)
        ```
    """

    def __init__(
        self,
        lipschitz_norm: float,
        backward_model: Union[BackwardModel, BackwardSequential],
        *args,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.lipschitz_norm = lipschitz_norm
        self.backward_model = backward_model

    def get_config(self):
        config = super().get_config()
        lipschitz_config = keras.saving.serialize_keras_object(self.lipschitz_norm)
        config["lipschitz_norm"] = lipschitz_config
        backward_config = keras.saving.serialize_keras_object(self.backward_model)
        config["backward_model"] = backward_config
        return config
