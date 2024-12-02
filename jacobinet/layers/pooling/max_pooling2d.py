import numpy as np
from keras.layers import Layer, Input, MaxPooling2D, Reshape
from keras.models import Sequential
import keras.ops as K
from jacobinet.layers.layer import BackwardNonLinearLayer
from jacobinet.layers.reshaping import BackwardReshape
from jacobinet.layers.convolutional import BackwardDepthwiseConv2D
from jacobinet.layers.custom.prime import max_prime
from .utils_conv import get_conv_op
from keras import KerasTensor as Tensor


class BackwardMaxPooling2D(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `MaxPooling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the max pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import MaxPooling2D
    from keras_custom.backward.layers import BackwardMaxPooling2D

    # Assume `max_pooling_layer` is a pre-defined MaxPooling2D layer
    backward_layer = BackwardMaxPooling2D(max_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: MaxPooling2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        conv_op, _ = get_conv_op(self.layer)
        self.conv_op = conv_op

        conv_out_shape_wo_batch = list(
            self.conv_op(Input(self.input_dim_wo_batch)).shape[1:]
        )
        if self.layer.data_format == "channels_first":

            in_channel = self.input_dim_wo_batch[0]
            image_shape = conv_out_shape_wo_batch[-2:]
            self.target_shape = [in_channel, -1] + image_shape
            self.axis = 2
        else:
            in_channel = self.input_dim_wo_batch[-1]
            image_shape = conv_out_shape_wo_batch[:2]
            self.target_shape = image_shape + [in_channel, -1]
            self.axis = -1

        self.reshape_op = Reshape(self.target_shape)
        # compile a model to init shape
        inner_model = Sequential([self.conv_op, self.reshape_op])
        _ = inner_model(Input(self.input_dim_wo_batch))

        self.backward_reshape_op = BackwardReshape(layer=self.reshape_op)
        self.backward_conv2d = BackwardDepthwiseConv2D(
            layer=self.conv_op
        )

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        inner_input = self.reshape_op(self.conv_op(input))

        backward_max = max_prime(inner_input, axis=self.axis)
        backward_max_fuse = backward_max * K.expand_dims(
            gradient, axis=self.axis
        )
        if self.layer.data_format == "channels_last":
            backward_max_fuse = K.transpose(backward_max_fuse, (0, 1, 2, 4, 3))
        backward_reshape = self.backward_reshape_op(backward_max_fuse)
        output = self.backward_conv2d(backward_reshape)
        return output


def get_backward_MaxPooling2D(layer: MaxPooling2D) -> Layer:
    """
    This function creates a `BackwardMaxPooling2D` layer based on a given `MaxPooling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `MaxPooling2D` layer, using the
    `BackwardMaxPooling2D` class to reverse the max pooling operation.

    ### Parameters:
    - `layer`: A Keras `MaxPooling2D` layer instance. The function uses this layer's configurations to set up the `BackwardMaxPooling2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMaxPooling2D`, which acts as the reverse layer for the given `MaxPooling2D`.

    ### Example Usage:
    ```python
    from keras.layers import MaxPooling2D
    from keras_custom.backward import get_backward_MaxPooling2D

    # Assume `maxpool_layer` is a pre-defined MaxPooling2D layer
    backward_layer = get_backward_MaxPooling2D(maxpool_layer)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardMaxPooling2D(layer)
    return layer_backward
