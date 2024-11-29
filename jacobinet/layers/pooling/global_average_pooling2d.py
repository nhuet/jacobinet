import numpy as np
import keras
from keras.layers import Layer, GlobalAveragePooling2D
import keras.ops as K
from jacobinet.layers.layer import BackwardLinearLayer


class BackwardGlobalAveragePooling2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `GlobalAveragePooling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the average pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import GlobalAveragePooling2D
    from keras_custom.backward.layers import BackwardGlobalAveragePooling2D

    # Assume `average_pooling_layer` is a pre-defined GlobalAveragePooling2D layer
    backward_layer = BackwardGlobalAveragePooling2D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        if self.layer.data_format == "channels_first":
            w_in, h_in = self.layer.input.shape[-2:]
            if self.layer.keepdims:
                output = K.repeat(K.repeat(gradient, w_in, -2), h_in, -1) / (
                    w_in * h_in
                )
            else:
                output = K.repeat(
                    K.repeat(
                        K.expand_dims(K.expand_dims(gradient, -1), -1),
                        w_in,
                        -2,
                    ),
                    h_in,
                    -1,
                ) / (w_in * h_in)
        else:
            w_in, h_in = self.layer.input.shape[1:3]
            if self.layer.keepdims:
                output = K.repeat(K.repeat(gradient, w_in, 1), h_in, 2) / (
                    w_in * h_in
                )
            else:
                output = K.repeat(
                    K.repeat(
                        K.expand_dims(K.expand_dims(gradient, 1), 1), w_in, 1
                    ),
                    h_in,
                    2,
                ) / (w_in * h_in)
        return output


def get_backward_GlobalAveragePooling2D(
    layer: GlobalAveragePooling2D
) -> Layer:
    """
    This function creates a `BackwardGlobalAveragePooling2D` layer based on a given `GlobalAveragePooling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `GlobalAveragePooling2D` layer, using the
    `BackwardGlobalAveragePooling2D` class to reverse the average pooling operation.

    ### Parameters:
    - `layer`: A Keras `GlobalAveragePooling2D` layer instance. The function uses this layer's configurations to set up the `BackwardGlobalAveragePooling2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardGlobalAveragePooling2D`, which acts as the reverse layer for the given `GlobalAveragePooling2D`.

    ### Example Usage:
    ```python
    from keras.layers import GlobalAveragePooling2D
    from keras_custom.backward import get_backward_GlobalAveragePooling2D

    # Assume `average_layer` is a pre-defined GlobalAveragePooling2D layer
    backward_layer = get_backward_GlobalAveragePooling2D(average_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardGlobalAveragePooling2D(layer)
    return layer_backward
