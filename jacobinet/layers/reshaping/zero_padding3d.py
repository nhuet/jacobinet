from keras.layers import ZeroPadding3D, Cropping3D
from keras.layers import Layer
import keras.ops as K
from jacobinet.layers.layer import BackwardLinearLayer


class BackwardZeroPadding3D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `ZeroPadding3D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the zero padding
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import ZeroPadding3D
    from keras_custom.backward.layers import BackwardZeroPadding3D

    # Assume `cropping_layer` is a pre-defined ZeroPadding3D layer
    backward_layer = BackwardZeroPadding3D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: ZeroPadding3D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_padding = layer.get_config()
        padding = dico_padding["padding"]
        data_format = dico_padding["data_format"]

        self.layer_backward = Cropping3D(
            cropping=padding, data_format=data_format
        )
        self.layer_backward.built = True


def get_backward_ZeroPadding3D(layer: ZeroPadding3D, use_bias=True) -> Layer:
    """
    This function creates a `BackwardZeroPadding3D` layer based on a given `ZeroPadding3D` layer. It provides
    a convenient way to obtain a backward approximation of the input `ZeroPadding3D` layer, using the
    `BackwardZeroPadding3D` class to reverse the zero padding operation.

    ### Parameters:
    - `layer`: A Keras `ZeroPadding3D` layer instance. The function uses this layer's configurations to set up the `BackwardZeroPadding3D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardZeroPadding3D`, which acts as the reverse layer for the given `ZeroPadding3D`.

    ### Example Usage:
    ```python
    from keras.layers import ZeroPadding3D
    from keras_custom.backward import get_backward_ZeroPadding3D

    # Assume `zero_padding_layer` is a pre-defined ZeroPadding3D layer
    backward_layer = get_backward_ZeroPadding3D(zero_padding_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    dico_padding = layer.get_config()
    padding = dico_padding["padding"]
    data_format = dico_padding["data_format"]

    layer_backward = Cropping3D(cropping=padding, data_format=data_format)
    layer_backward.built = True

    return BackwardZeroPadding3D(layer)
