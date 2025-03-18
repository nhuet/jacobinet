import keras
from keras.layers import UpSampling2D  # type: ignore
from keras.layers import Layer  # type: ignore
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


@keras.saving.register_keras_serializable()
class BackwardUpSampling2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `UpSampling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the upsampling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import UpSampling2D
    from keras_custom.backward.layers import BackwardUpSampling2D

    # Assume `upsampling_layer` is a pre-defined UpSampling2D layer
    backward_layer = BackwardUpSampling2D(upsampling_layer)
    output = backward_layer(input_tensor)
    """

    layer: UpSampling2D

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        # If data_format is "channels_last": (batch_size, rows, cols, channels)
        if self.layer.interpolation != "nearest":
            raise NotImplementedError(
                "actually not working wih interpolation in {'bicubic', 'bilinear', 'lanczos3', 'lanczos5'}. Raise a dedicated PR if needed"
            )
        if self.layer.data_format == "channels_last":
            # (batch_size, W, H, C)
            W, H, C = self.input_dim_wo_batch
            gradient = K.sum(
                K.reshape(
                    gradient,
                    [-1, W, self.layer.size[0], H, self.layer.size[1], C],
                ),
                (2, 4),
            )
        else:
            C, W, H = self.input_dim_wo_batch
            gradient = K.sum(
                K.reshape(
                    gradient,
                    [-1, C, W, self.layer.size[0], H, self.layer.size[1]],
                ),
                (3, 5),
            )

        return gradient


def get_backward_UpSampling2D(layer: UpSampling2D) -> Layer:
    """
    This function creates a `BackwardUpSampling2D` layer based on a given `UpSampling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `UpSampling2D` layer, using the
    `BackwardUpSampling2D` class to reverse the upsampling operation.

    ### Parameters:
    - `layer`: A Keras `UpSampling2D` layer instance. The function uses this layer's configurations to set up the `BackwardUpSampling2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardUpSampling2D`, which acts as the reverse layer for the given `UpSampling2D`.

    ### Example Usage:
    ```python
    from keras.layers import UpSampling2D
    from keras_custom.backward import get_backward_UpSampling2D

    # Assume `upsampling_layer` is a pre-defined UpSampling2D layer
    backward_layer = get_backward_UpSampling2D(zero_padding_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardUpSampling2D(layer)
