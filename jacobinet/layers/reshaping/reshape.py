from keras.layers import Reshape  # type: ignore
from keras.layers import Layer  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


class BackwardReshape(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Reshape` layer in Keras.
    It can be used to apply operations in a reverse manner, transposing the permute
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Reshape
    from keras_custom.backward.layers import BackwardReshape

    # Assume `reshape_layer` is a pre-defined Reshape layer
    backward_layer = BackwardReshape(permute_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Reshape,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        self.layer_backward = Reshape(target_shape=self.input_dim_wo_batch)


def get_backward_Reshape(layer: Reshape) -> Layer:
    """
    This function creates a `BackwardReshape` layer based on a given `Reshape` layer. It provides
    a convenient way to obtain a backward approximation of the input `Reshape` layer, using the
    `BackwardReshape` class to reverse the reshape operation.

    ### Parameters:
    - `layer`: A Keras `Reshape` layer instance. The function uses this layer's configurations to set up the `BackwardReshape` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardReshape`, which acts as the reverse layer for the given `Reshape`.

    ### Example Usage:
    ```python
    from keras.layers import Reshape
    from keras_custom.backward import get_backward_Reshape

    # Assume `reshape_layer` is a pre-defined Reshape layer
    backward_layer = get_backward_Reshape(reshape_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardReshape(layer)
