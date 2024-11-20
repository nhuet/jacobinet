from keras.layers import Layer, DepthwiseConv1D, Conv1DTranspose, Reshape
from jacobinet.layers.layer import BackwardLinearLayer
from jacobinet.layers.utils import pooling_layer1D
from keras.models import Sequential
import keras.ops as K

from typing import List


class BackwardDepthwiseConv1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `DepthwiseConv1D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the depthwise convolution
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv1D
    from keras_custom.backward.layers import BackwardDepthwiseConv1D

    # Assume `depthwise_conv_layer` is a pre-defined DepthwiseConv1D layer
    backward_layer = BackwardDepthwiseConv1D(depthwise_conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: DepthwiseConv1D,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

        # input_dim_wo_batch = self.layer.input.shape[1:]
        # output_dim_wo_batch = self.layer.output.shape[1:]
        input_dim_wo_batch = self.input_dim_wo_batch
        output_dim_wo_batch = self.output_dim_wo_batch

        self.d_m = self.layer.depth_multiplier
        if self.layer.data_format == "channels_first":
            c_in = input_dim_wo_batch[0]
            w_out = output_dim_wo_batch[-1]
            target_shape = [c_in, self.layer.depth_multiplier, w_out]

            split_shape = [self.layer.depth_multiplier, w_out]
            self.axis = 1
            self.c_in = c_in
            self.axis_c = 1
        else:
            c_in = input_dim_wo_batch[-1]
            w_out = output_dim_wo_batch[0]
            target_shape = [w_out, c_in, self.layer.depth_multiplier]
            split_shape = [w_out, self.layer.depth_multiplier]
            self.axis = -1
            self.c_in = c_in
            self.axis_c = -2

        self.op_reshape = Reshape(target_shape)
        self.op_split = Reshape(split_shape)

        # c_in convolution operator

        conv_transpose_list: List[Conv1DTranspose] = []

        for i in range(c_in):
            kernel_i = self.layer.kernel[:, i : i + 1]  # (kernel_w, 1, d_m)
            dico_depthwise_conv = layer.get_config()
            dico_depthwise_conv["filters"] = dico_depthwise_conv[
                "depth_multiplier"
            ]
            dico_depthwise_conv["kernel_initializer"] = dico_depthwise_conv[
                "depthwise_initializer"
            ]
            dico_depthwise_conv["kernel_regularizer"] = dico_depthwise_conv[
                "depthwise_regularizer"
            ]
            dico_depthwise_conv["kernel_constraint"] = dico_depthwise_conv[
                "depthwise_constraint"
            ]
            dico_depthwise_conv["padding"] = "valid"

            # remove unknown features in Conv1DTranspose
            dico_depthwise_conv.pop("depth_multiplier")
            dico_depthwise_conv.pop("depthwise_initializer")
            dico_depthwise_conv.pop("depthwise_regularizer")
            dico_depthwise_conv.pop("depthwise_constraint")

            dico_depthwise_conv["use_bias"] = False

            conv_t_i = Conv1DTranspose.from_config(dico_depthwise_conv)
            conv_t_i.kernel = kernel_i
            conv_t_i.built = True
            conv_transpose_list.append(conv_t_i)

        # shape of transposed input
        # input_shape_wo_batch = list(layer.input.shape[1:])
        input_shape_wo_batch = self.input_dim_wo_batch
        input_shape_wo_batch_wo_pad = list(
            (
                K.repeat(
                    conv_t_i(K.zeros([1] + split_shape)), c_in, axis=self.axis
                )[0]
            ).shape
        )

        if layer.data_format == "channels_first":
            w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
        else:
            w_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]

        pad_layers = pooling_layer1D(w_pad, self.layer.data_format)
        if len(pad_layers):
            self.inner_models = [
                Sequential([conv_t_i] + pad_layers)
                for conv_t_i in conv_transpose_list
            ]
        else:
            self.inner_models = conv_transpose_list

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        # remove bias if needed
        if self.layer.use_bias and self.use_bias:
            if self.layer.data_format == "channels_first":
                gradient = (
                    gradient - self.layer.bias[None, :, None]
                )  # (batch, d_m*c_in, w_out)
            else:
                gradient = (
                    gradient - self.layer.bias[None, None, :]
                )  # (batch, w_out, d_m*c_in)
        outputs = self.op_reshape(
            gradient
        )  # (batch, d_m, c_in, w_out) if data_format=channel_first

        split_outputs = K.split(
            outputs, self.c_in, axis=self.axis_c
        )  # [(batch, d_m, 1, w_out, h_out)]
        split_outputs = [
            self.op_split(s_o_i) for s_o_i in split_outputs
        ]  # [(batch_size, d_m, w_out, h_out)]

        conv_outputs = [
            self.inner_models[i](s_o_i)
            for (i, s_o_i) in enumerate(split_outputs)
        ]  # [(batch_size, 1, w_in, h_in)]
        output = K.concatenate(
            conv_outputs, axis=self.axis
        )  # (batch_size, c_in, w_in, h_in)
        return output


def get_backward_DepthwiseConv1D(
    layer: DepthwiseConv1D, use_bias=True
) -> Layer:
    """
    This function creates a `BackwardDepthwiseConv1D` layer based on a given `DepthwiseConv1D` layer. It provides
    a convenient way to obtain a backward pass of the input `DepthwiseConv1D` layer, using the
    `BackwardDepthwiseConv1D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `DepthwiseConv1D` layer instance. The function uses this layer's configurations (input and output shapes,
      depth multiplier, data format) to set up the `BackwardDepthwiseConv1D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardDepthwiseConv1D`, which acts as the reverse layer for the given `DepthwiseConv1D`.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv1D
    from keras_custom.layers import get_backward_DepthwiseConv1D

    # Assume `depthwise_conv_layer` is a pre-defined DepthwiseConv1D layer
    backward_layer = get_backward_DepthwiseConv1D(depthwise_conv_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    layer_backward = BackwardDepthwiseConv1D(layer, use_bias)
    return layer_backward
