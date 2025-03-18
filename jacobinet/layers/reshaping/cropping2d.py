import keras
from jacobinet.layers.layer import BackwardLinearLayer
from keras.layers import Layer  # type: ignore
from keras.layers import Cropping2D, ZeroPadding2D  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardCropping2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Cropping2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the cropping
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Cropping2D
    from keras_custom.backward.layers import BackwardCropping2D

    # Assume `cropping_layer` is a pre-defined Cropping2D layer
    backward_layer = BackwardCropping2D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Cropping2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_cropping = layer.get_config()
        cropping = dico_cropping["cropping"]
        data_format = dico_cropping["data_format"]

        self.layer_backward = ZeroPadding2D(padding=cropping, data_format=data_format)
        self.layer_backward.built = True


def get_backward_Cropping2D(layer: Cropping2D) -> Layer:
    """
    This function creates a `BackwardCropping2D` layer based on a given `Cropping2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `Cropping2D` layer, using the
    `BackwardCropping2D` class to reverse the cropping operation.

    ### Parameters:
    - `layer`: A Keras `Cropping2D` layer instance. The function uses this layer's configurations to set up the `BackwardCropping2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardCropping2D`, which acts as the reverse layer for the given `Cropping2D`.

    ### Example Usage:
    ```python
    from keras.layers import Cropping2D
    from keras_custom.backward import get_backward_Cropping2D

    # Assume `cropping_layer` is a pre-defined Cropping2D layer
    backward_layer = get_backward_Cropping2D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardCropping2D(layer)
