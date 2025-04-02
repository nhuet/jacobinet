import keras
import keras.ops as K  # type: ignore
import numpy as np
from jacobinet.layers.custom.prime import max_prime
from jacobinet.layers.layer import BackwardNonLinearLayer
from jacobinet.layers.utils import reshape_to_batch
from keras.layers import GlobalMaxPooling2D, Layer  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardGlobalMaxPooling2D(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `GlobalMaxPooling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the global max pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import GlobalMaxPooling2D
    from keras_custom.backward.layers import BackwardGlobalMaxPooling2D

    # Assume `max_pooling_layer` is a pre-defined GlobalMaxPooling2D layer
    backward_layer = BackwardGlobalMaxPooling2D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    layer: GlobalMaxPooling2D

    def call(self, inputs, training=None, mask=None):
        gradient = inputs[0]  # (batch, n_out..., C_W_out, H_out)

        layer_input = inputs[1]  # (batch, C, W_in, H_in)

        reshape_tag, gradient_, n_out = reshape_to_batch(gradient, [1] + self.output_dim_wo_batch)
        if not self.layer.keepdims:
            if self.layer.data_format == "channels_first":
                gradient_ = K.expand_dims(K.expand_dims(gradient, -1), -1)
            else:
                gradient_ = K.expand_dims(K.expand_dims(gradient, 1), 1)
        # gradient_ (batch*prod(n_out), output_dim_wo_batch...)= (batch*prod(n_out), C, W_out, H_out) if channel_first

        if self.layer.data_format == "channels_first":
            axis = -1
            C, W, H = self.input_dim_wo_batch
            layer_input_ = K.reshape(layer_input, [-1, C, W * H])
            backward_max = K.reshape(
                max_prime(layer_input_, axis=axis), [-1, C, W, H]
            )  # (batch, C, W_in*H_in)
        else:
            axis = 1
            W, H, C = self.input_dim_wo_batch
            layer_input_ = K.reshape(layer_input, [-1, W * H, C])
            backward_max = K.reshape(max_prime(layer_input_, axis=axis), [-1, W, H, C])

        # combine backward_max with gradient, we first need to reshape gradient_
        if len(n_out):
            gradient_ = K.reshape(
                gradient_, [-1, np.prod(n_out)] + self.output_dim_wo_batch
            )  # (batch, N_out, C, W_out, H_out)

        # n_out=[] gradient_.shape = (batch, N_out, C, W_out, H_out) else (batch, C, W_out, H_out)
        if len(n_out):
            backward_max = K.expand_dims(backward_max, 1)  # (batch, 1, C,W_out, H_out)

        # element wise product to apply chain rule
        output = (
            gradient_ * backward_max
        )  # (batch, N_out, C, W_out, H_out) or (batch, C, W_out, H_out)

        # reshape_tag
        if reshape_tag:
            output = K.reshape(output, [-1] + n_out + self.input_dim_wo_batch)

        return output


def get_backward_GlobalMaxPooling2D(layer: GlobalMaxPooling2D) -> Layer:
    """
    This function creates a `BackwardGlobalMaxPooling2D` layer based on a given `GlobalMaxPooling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `GlobalMaxPooling2D` layer, using the
    `BackwardGlobalMaxPooling2D` class to reverse the max pooling operation.

    ### Parameters:
    - `layer`: A Keras `GlobalMaxPooling2D` layer instance. The function uses this layer's configurations to set up the `BackwardGlobalMaxPooling2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardGlobalMaxPooling2D`, which acts as the reverse layer for the given `GlobalMaxPooling2D`.

    ### Example Usage:
    ```python
    from keras.layers import GlobalMaxPooling2D
    from keras_custom.backward import get_backward_GlobalMaxPooling2D

    # Assume `maxpool_layer` is a pre-defined GlobalMaxPooling2D layer
    backward_layer = get_backward_GlobalMaxPooling2D(maxpool_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardGlobalMaxPooling2D(layer)
    return layer_backward
