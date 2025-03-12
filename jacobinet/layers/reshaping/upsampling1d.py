from keras.layers import UpSampling1D  # type: ignore
from keras.layers import Layer  # type: ignore
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


class BackwardUpSampling1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `UpSampling1D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the upsampling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import UpSampling1D
    from keras_custom.backward.layers import BackwardUpSampling1D

    # Assume `upsampling_layer` is a pre-defined UpSampling1D layer
    backward_layer = BackwardUpSampling1D(upsampling_layer)
    output = backward_layer(input_tensor)
    """

    layer: UpSampling1D

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        # (batch_size, steps, feature)
        steps, features = self.input_dim_wo_batch
        gradient = K.sum(
            K.reshape(gradient, [-1, steps, self.layer.size, features]), 2
        )
        return gradient


def get_backward_UpSampling1D(layer: UpSampling1D) -> Layer:
    """
    This function creates a `BackwardUpSampling1D` layer based on a given `UpSampling1D` layer. It provides
    a convenient way to obtain a backward approximation of the input `UpSampling1D` layer, using the
    `BackwardUpSampling1D` class to reverse the upsampling operation.

    ### Parameters:
    - `layer`: A Keras `UpSampling1D` layer instance. The function uses this layer's configurations to set up the `BackwardUpSampling1D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardUpSampling1D`, which acts as the reverse layer for the given `UpSampling1D`.

    ### Example Usage:
    ```python
    from keras.layers import UpSampling1D
    from keras_custom.backward import get_backward_UpSampling1D

    # Assume `upsampling_layer` is a pre-defined UpSampling1D layer
    backward_layer = get_backward_UpSampling1D(zero_padding_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardUpSampling1D(layer)
