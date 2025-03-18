import keras
from keras.layers import ZeroPadding1D, Cropping1D  # type: ignore
from keras.layers import Layer  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


@keras.saving.register_keras_serializable()
class BackwardCropping1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Cropping1D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the cropping
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Cropping1D
    from keras_custom.backward.layers import BackwardCropping1D

    # Assume `cropping_layer` is a pre-defined Cropping1D layer
    backward_layer = BackwardCropping1D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Cropping1D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_cropping = layer.get_config()
        cropping = dico_cropping["cropping"]
        # axis = 1
        # warning of cropping1D: Input shape is a 3D tensor with shape (batch_size, axis_to_crop, features)
        self.layer_backward = ZeroPadding1D(
            padding=cropping, data_format="channels_last"
        )
        self.layer_backward.built = True


def get_backward_Cropping1D(layer: Cropping1D) -> Layer:
    """
    This function creates a `BackwardCropping1D` layer based on a given `Cropping1D` layer. It provides
    a convenient way to obtain a backward approximation of the input `Cropping1D` layer, using the
    `BackwardCropping1D` class to reverse the cropping operation.

    ### Parameters:
    - `layer`: A Keras `Cropping1D` layer instance. The function uses this layer's configurations to set up the `BackwardCropping1D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardCropping1D`, which acts as the reverse layer for the given `Cropping1D`.

    ### Example Usage:
    ```python
    from keras.layers import Cropping1D
    from keras_custom.backward import get_backward_Cropping1D

    # Assume `cropping_layer` is a pre-defined Cropping1D layer
    backward_layer = get_backward_Cropping1D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardCropping1D(layer)
