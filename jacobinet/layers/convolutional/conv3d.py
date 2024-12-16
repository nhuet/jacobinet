from keras.layers import Conv3D, Conv3DTranspose, Input, Activation
from keras.layers import Layer
from keras.models import Sequential
import keras.ops as K
from jacobinet.layers.utils import pooling_layer3D
from jacobinet.layers.layer import BackwardLinearLayer, BackwardNonLinearLayer
from jacobinet.layers.core.activations import BackwardActivation

from keras import KerasTensor as Tensor


class BackwardConv3D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Conv3D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv3D
    from keras_custom.backward.layers import BackwardConv3D

    # Assume `conv_layer` is a pre-defined Conv3D layer
    backward_layer = BackwardConv3D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv3D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_conv = layer.get_config()
        dico_conv.pop("groups")
        # input_shape = list(layer.input.shape[1:])
        input_shape = self.input_dim_wo_batch
        # update filters to match input, pay attention to data_format
        if (
            layer.data_format == "channels_first"
        ):  # better to use enum than raw str
            dico_conv["filters"] = input_shape[0]
        else:
            dico_conv["filters"] = input_shape[-1]

        dico_conv["use_bias"] = False
        dico_conv["padding"] = "valid"

        layer_backward = Conv3DTranspose.from_config(dico_conv)
        layer_backward.kernel = layer.kernel

        layer_backward.built = True

        # input_shape_wo_batch = list(layer.input.shape[1:])
        input_shape_wo_batch = self.input_dim_wo_batch
        # input_shape_wo_batch_wo_pad = list(layer_backward(layer.output)[0].shape)
        input_shape_wo_batch_wo_pad = list(
            layer_backward(K.ones([1] + self.output_dim_wo_batch))[0].shape
        )

        if layer.data_format == "channels_first":
            d_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
            w_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]
            h_pad = input_shape_wo_batch[3] - input_shape_wo_batch_wo_pad[3]
        else:
            d_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]
            w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
            h_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]

        pad_layers = pooling_layer3D(d_pad, w_pad, h_pad, layer.data_format)
        if len(pad_layers):
            layer_backward = Sequential([layer_backward] + pad_layers)
            layer_backward(Input(self.output_dim_wo_batch))
            # _ = layer_backward([1]+self.output_dim_wo_batch)
        self.layer_backward = layer_backward


class BackwardConv3DWithActivation(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Conv3D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv3D
    from keras_custom.backward.layers import BackwardConv1D

    # Assume `conv_layer` is a pre-defined Conv3D layer with an activation function
    backward_layer = BackwardConv3DWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv3D,
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
        self.layer_wo_activation = Conv3D.from_config(dico_config)
        self.layer_wo_activation._kernel = self.layer.kernel
        self.layer_wo_activation.bias = self.layer.bias
        self.layer_wo_activation.built=True
        self.layer_backward = BackwardConv3D(self.layer_wo_activation, 
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


def get_backward_Conv3D(layer: Conv3D) -> Layer:
    """
    This function creates a `BackwardConv3D` layer based on a given `Conv3D` layer. It provides
    a convenient way to obtain the backward pass of the input `Conv3D` layer, using the
    `BackwardConv3D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `Conv3D` layer instance. The function uses this layer's configurations to set up the `BackwardConv3D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardConv3D`, which acts as the reverse layer for the given `Conv3D`.

    ### Example Usage:
    ```python
    from keras.layers import Conv3D
    from keras_custom.backward import get_backward_Conv3D

    # Assume `conv_layer` is a pre-defined Conv3D layer
    backward_layer = get_backward_Conv3D(conv_layer)
    output = backward_layer(input_tensor)
    """
    if layer.get_config()['activation']=='linear':
        return BackwardConv3D(layer)
    else:
        return BackwardConv3DWithActivation(layer)


