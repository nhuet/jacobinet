from keras.layers import Layer, DepthwiseConv2D, Conv2DTranspose, Reshape, Activation
from keras.models import Sequential
import keras.ops as K
from jacobinet.layers.layer import BackwardLinearLayer, BackwardNonLinearLayer
from jacobinet.layers.core.activations import BackwardActivation
from jacobinet.layers.utils import pooling_layer2D, call_backward_depthwise2d

from keras import KerasTensor as Tensor
from typing import List


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
            kernel_i = self.layer.kernel[
                :, :, i : i + 1
            ]  # (kernel_w, kernel_h, 1, d_m)
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
            K.repeat(
                conv_t_i(K.zeros([1] + split_shape)), c_in, axis=self.axis
            )[0]
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
                Sequential([conv_t_i] + pad_layers)
                for conv_t_i in conv_transpose_list
            ]
        else:
            self.inner_models = conv_transpose_list

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
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

class BackwardDepthwiseConv1DWithActivation(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `DepthwiseConv1D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv1
    from keras_custom.backward.layers import BackwardConv1D

    # Assume `conv_layer` is a pre-defined Conv1D layer with an activation function
    backward_layer = BackwardConv1DDWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: DepthwiseConv2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        activation_name = layer.get_config()["activation"]
        self.activation_backward = BackwardActivation(Activation(activation_name), 
                                                      input_dim_wo_batch = self.output_dim_wo_batch,
                                                      output_dim_wo_batch = self.output_dim_wo_batch)
        
        #deserialize(activation_name)

        dico_config = self.layer.get_config()
        dico_config['activation']='linear'
        self.layer_wo_activation = DepthwiseConv2D.from_config(dico_config)
        #self.layer_wo_activation._kernel = self.layer._kernel
        #self.layer_wo_activation.bias = self.layer.bias
        self.layer_wo_activation.built=True
        self.layer_backward = BackwardDepthwiseConv2D(self.layer_wo_activation, 
                                             input_dim_wo_batch= self.input_dim_wo_batch, 
                                             output_dim_wo_batch = self.output_dim_wo_batch)
        
        self.layer_wo_activation.built=True

    def call(self, inputs, training=None, mask=None):
         # apply locally the chain rule
        # (f(g(x)))' = f'(x)*g'(f(x))
        # compute f(x) as inner_input
        
        gradient = inputs[0]
        input = inputs[1]
        inner_input = self.layer_wo_activation(input)
        # computer gradient*g'(f(x))
        backward_output: Tensor = self.activation_backward(inputs=[gradient, inner_input])
        # compute gradient*g'(f(x))*f'(x)
        output = self.layer_backward(inputs=[backward_output])

        return output
    
def get_backward_DepthwiseConv2D(
    layer: DepthwiseConv2D
) -> Layer:
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
    if layer.get_config()['activation']=='linear':
        return BackwardDepthwiseConv2D(layer)
    else:
        return BackwardDepthwiseConv2DWithActivation(layer)

