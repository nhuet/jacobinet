from keras.layers import Layer, GlobalAveragePooling3D  # type: ignore
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


class BackwardGlobalAveragePooling3D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `GlobalAveragePooling3D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the average pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import GlobalAveragePooling3D
    from keras_custom.backward.layers import BackwardGlobalAveragePooling3D

    # Assume `average_pooling_layer` is a pre-defined GlobalAveragePooling3D layer
    backward_layer = BackwardGlobalAveragePooling3D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    layer: GlobalAveragePooling3D

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        if self.layer.data_format == "channels_first":
            d_in, w_in, h_in = self.layer.input.shape[-3:]
            if self.layer.keepdims:
                output = K.repeat(
                    K.repeat(K.repeat(gradient, d_in, -3), w_in, -2),
                    h_in,
                    -1,
                ) / (w_in * h_in * d_in)
            else:
                output = K.repeat(
                    K.repeat(
                        K.repeat(
                            K.expand_dims(
                                K.expand_dims(K.expand_dims(gradient, -1), -1),
                                -1,
                            ),
                            d_in,
                            -3,
                        ),
                        w_in,
                        -2,
                    ),
                    h_in,
                    -1,
                ) / (w_in * h_in * d_in)
        else:
            d_in, w_in, h_in = self.layer.input.shape[1:4]
            if self.layer.keepdims:
                output = K.repeat(
                    K.repeat(K.repeat(gradient, d_in, 1), w_in, 2),
                    h_in,
                    3,
                ) / (w_in * h_in * d_in)
            else:
                output = K.repeat(
                    K.repeat(
                        K.repeat(
                            K.expand_dims(
                                K.expand_dims(K.expand_dims(gradient, 1), 1), 1
                            ),
                            d_in,
                            1,
                        ),
                        w_in,
                        2,
                    ),
                    h_in,
                    3,
                ) / (w_in * h_in * d_in)
        return output


def get_backward_GlobalAveragePooling3D(
    layer: GlobalAveragePooling3D,
) -> Layer:
    """
    This function creates a `BackwardGlobalAveragePooling3D` layer based on a given `GlobalAveragePooling3D` layer. It provides
    a convenient way to obtain a backward approximation of the input `GlobalAveragePooling3D` layer, using the
    `BackwardGlobalAveragePooling3D` class to reverse the average pooling operation.

    ### Parameters:
    - `layer`: A Keras `GlobalAveragePooling3D` layer instance. The function uses this layer's configurations to set up the `BackwardGlobalAveragePooling3D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardGlobalAveragePooling3D`, which acts as the reverse layer for the given `GlobalAveragePooling3D`.

    ### Example Usage:
    ```python
    from keras.layers import GlobalAveragePooling3D
    from keras_custom.backward import get_backward_GlobalAveragePooling3D

    # Assume `average_layer` is a pre-defined GlobalAveragePooling3D layer
    backward_layer = get_backward_GlobalAveragePooling3D(average_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardGlobalAveragePooling3D(layer)
    return layer_backward
