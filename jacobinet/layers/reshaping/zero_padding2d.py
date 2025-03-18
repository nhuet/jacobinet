import keras
from keras.layers import ZeroPadding2D, Cropping2D  # type: ignore
from keras.layers import Layer  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


@keras.saving.register_keras_serializable()
class BackwardZeroPadding2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `ZeroPadding2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the zero padding
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import ZeroPadding2D
    from keras_custom.backward.layers import BackwardZeroPadding2D

    # Assume `cropping_layer` is a pre-defined ZeroPadding2D layer
    backward_layer = BackwardZeroPadding2D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: ZeroPadding2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_padding = layer.get_config()
        padding = dico_padding["padding"]
        data_format = dico_padding["data_format"]

        self.layer_backward = Cropping2D(
            cropping=padding, data_format=data_format
        )
        self.layer_backward.built = True


def get_backward_ZeroPadding2D(layer: ZeroPadding2D) -> Layer:
    """
    This function creates a `BackwardZeroPadding2D` layer based on a given `ZeroPadding2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `ZeroPadding2D` layer, using the
    `BackwardZeroPadding2D` class to reverse the zero padding operation.

    ### Parameters:
    - `layer`: A Keras `ZeroPadding2D` layer instance. The function uses this layer's configurations to set up the `BackwardZeroPadding2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardZeroPadding2D`, which acts as the reverse layer for the given `ZeroPadding2D`.

    ### Example Usage:
    ```python
    from keras.layers import ZeroPadding2D
    from keras_custom.backward import get_backward_ZeroPadding2D

    # Assume `zero_padding_layer` is a pre-defined ZeroPadding2D layer
    backward_layer = get_backward_ZeroPadding2D(zero_padding_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardZeroPadding2D(layer)
