from typing import List

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.core.activations import BackwardActivation
from jacobinet.layers.layer import BackwardLinearLayer, BackwardWithActivation
from jacobinet.layers.utils import call_backward_depthwise2d, pooling_layer2D
from keras import KerasTensor as Tensor
from keras.layers import (  # type: ignore
    Conv2DTranspose,
    DepthwiseConv2D,
    Layer,
    Reshape,
)
from keras.models import Sequential  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardDepthwiseConv2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `DepthwiseConv2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the depthwise convolution
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv2D
    from keras_custom.backward.layers import BackwardDepthwiseConv2D

    # Assume `depthwise_conv_layer` is a pre-defined DepthwiseConv2D layer
    backward_layer = BackwardDepthwiseConv2D(depthwise_conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: DepthwiseConv2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

        # input_dim_wo_batch = self.layer.input.shape[1:]
        input_dim_wo_batch = self.input_dim_wo_batch
        # output_dim_wo_batch = self.layer.output.shape[1:]
        output_dim_wo_batch = self.output_dim_wo_batch
        self.d_m = self.layer.depth_multiplier
        if self.layer.data_format == "channels_first":
            c_in = input_dim_wo_batch[0]
            w_out, h_out = output_dim_wo_batch[-2:]
            # target_shape = [self.layer.depth_multiplier, c_in, w_out, h_out]
            target_shape = [c_in, self.layer.depth_multiplier, w_out, h_out]

            split_shape = [self.layer.depth_multiplier, w_out, h_out]
            self.axis = 1
            self.c_in = c_in
            # self.axis_c = 2
            self.axis_c = 1
        else:
            c_in = input_dim_wo_batch[-1]
            w_out, h_out = output_dim_wo_batch[:2]
            # target_shape = [w_out, h_out, c_in, self.layer.depth_multiplier]
            target_shape = [w_out, h_out, self.layer.depth_multiplier, c_in]

            split_shape = [w_out, h_out, self.layer.depth_multiplier]
            self.axis = -1
            self.c_in = c_in
            # self.axis_c = -2
            self.axis_c = -1

        self.op_reshape = Reshape(target_shape)
        self.op_split = Reshape(split_shape)

        # c_in convolution operator

        conv_transpose_list: List[Conv2DTranspose] = []

        for i in range(c_in):
            kernel_i = self.layer.kernel[:, :, i : i + 1]  # (kernel_w, kernel_h, 1, d_m)
            dico_depthwise_conv = layer.get_config()
            dico_depthwise_conv["filters"] = dico_depthwise_conv["depth_multiplier"]
            dico_depthwise_conv["kernel_initializer"] = dico_depthwise_conv["depthwise_initializer"]
            dico_depthwise_conv["kernel_regularizer"] = dico_depthwise_conv["depthwise_regularizer"]
            dico_depthwise_conv["kernel_constraint"] = dico_depthwise_conv["depthwise_constraint"]
            dico_depthwise_conv["padding"] = "valid"  # self.layer.padding

            # remove unknown features in Conv2DTranspose
            dico_depthwise_conv.pop("depth_multiplier")
            dico_depthwise_conv.pop("depthwise_initializer")
            dico_depthwise_conv.pop("depthwise_regularizer")
            dico_depthwise_conv.pop("depthwise_constraint")

            dico_depthwise_conv["use_bias"] = False

            conv_t_i = Conv2DTranspose.from_config(dico_depthwise_conv)
            conv_t_i.kernel = kernel_i
            conv_t_i.built = True
            conv_transpose_list.append(conv_t_i)

        # shape of transposed input
        input_dim_wo_batch_t = (
            K.repeat(conv_t_i(K.zeros([1] + split_shape)), c_in, axis=self.axis)[0]
        ).shape
        if self.layer.data_format == "channels_first":
            w_pad = input_dim_wo_batch[-2] - input_dim_wo_batch_t[-2]
            h_pad = input_dim_wo_batch[-1] - input_dim_wo_batch_t[-1]
        else:
            w_pad = input_dim_wo_batch[0] - input_dim_wo_batch_t[0]
            h_pad = input_dim_wo_batch[1] - input_dim_wo_batch_t[1]

        pad_layers = pooling_layer2D(w_pad, h_pad, layer.data_format)
        if len(pad_layers):
            self.inner_models = [
                Sequential([conv_t_i] + pad_layers) for conv_t_i in conv_transpose_list
            ]
        else:
            self.inner_models = conv_transpose_list

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        output = call_backward_depthwise2d(
            gradient,
            self.layer,
            self.op_reshape,
            self.op_split,
            self.inner_models,
            self.axis,
            self.axis_c,
            self.c_in,
            False,
        )
        return output

@keras.saving.register_keras_serializable()
class BackwardDepthwiseConv2DWithActivation(BackwardWithActivation):
    """
    This class implements a custom layer for backward pass of a `DepthwiseConv2D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv2D
    from keras_custom.backward.layers import BackwardDepthwiseConv2DWithActivatio

    # Assume `conv_layer` is a pre-defined DepthwiseConv1D layer with an activation function
    backward_layer = BackwardDepthwiseConv2DWithActivatio(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: DepthwiseConv2D,
        **kwargs,
    ):
        super().__init__(
            layer=layer,
            backward_linear=BackwardDepthwiseConv2D,
            backward_activation=BackwardActivation,
            **kwargs,
        )


def get_backward_DepthwiseConv2D(layer: DepthwiseConv2D) -> Layer:
    """
    This function creates a `BackwardDepthwiseConv2D` layer based on a given `DepthwiseConv2D` layer. It provides
    a convenient way to obtain a backward pass of the input `DepthwiseConv2D` layer, using the
    `BackwardDepthwiseConv2D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `DepthwiseConv2D` layer instance. The function uses this layer's configurations (input and output shapes,
      depth multiplier, data format) to set up the `BackwardDepthwiseConv2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardDepthwiseConv2D`, which acts as the reverse layer for the given `DepthwiseConv2D`.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv2D
    from keras_custom.backward import get_backward_DepthwiseConv2D

    # Assume `depthwise_conv_layer` is a pre-defined DepthwiseConv2D layer
    backward_layer = get_backward_DepthwiseConv2D(depthwise_conv_layer)
    output = backward_layer(input_tensor)
    """
    if layer.get_config()["activation"] == "linear":
        return BackwardDepthwiseConv2D(layer)
    else:
        return BackwardDepthwiseConv2DWithActivation(layer)
