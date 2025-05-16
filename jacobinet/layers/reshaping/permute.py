from typing import Any

import keras
import numpy as np
from jacobinet.layers.layer import BackwardLinearLayer
from keras.layers import Layer  # type: ignore
from keras.layers import Permute  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardPermute(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Permute` layer in Keras.
    It can be used to apply operations in a reverse manner, transposing the permute
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Permute
    from keras_custom.backward.layers import BackwardPermute

    # Assume `permute_layer` is a pre-defined Permute layer
    backward_layer = BackwardPermute(permute_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Permute,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        dims = layer.dims
        dims_invert = np.argsort(dims) + 1
        self.layer_backward = Permute(dims=dims_invert)


def get_backward_Permute(layer: Permute) -> Layer:
    """
    This function creates a `BackwardPermute` layer based on a given `Permute` layer. It provides
    a convenient way to obtain a backward approximation of the input `Permute` layer, using the
    `BackwardPermute` class to reverse the permute operation.

    ### Parameters:
    - `layer`: A Keras `Permute` layer instance. The function uses this layer's configurations to set up the `BackwardPermute` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardPermute`, which acts as the reverse layer for the given `Permute`.

    ### Example Usage:
    ```python
    from keras.layers import Permute
    from keras_custom.backward import get_backward_Permute

    # Assume `permute_layer` is a pre-defined Permute layer
    backward_layer = get_backward_Permute(permute_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardPermute(layer)
