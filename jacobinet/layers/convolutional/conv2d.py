from keras.layers import Conv2D, Conv2DTranspose, Input, Activation
from keras.layers import Layer
from keras.models import Sequential
import keras.ops as K
from jacobinet.layers.utils import pooling_layer2D
from jacobinet.layers.layer import BackwardLinearLayer, BackwardWithActivation
from jacobinet.layers.core.activations import BackwardActivation

from keras import KerasTensor as Tensor

class BackwardConv2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Conv2D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv2D
    from keras_custom.backward.layers import BackwardConv2D

    # Assume `conv_layer` is a pre-defined Conv2D layer
    backward_layer = BackwardConv2D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv2D,
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

        layer_backward = Conv2DTranspose.from_config(dico_conv)
        layer_backward.kernel = layer.kernel

        layer_backward.built = True

        # input_shape_wo_batch = list(layer.input.shape[1:])
        input_shape_wo_batch = self.input_dim_wo_batch
        # input_shape_wo_batch_wo_pad = list(layer_backward(layer.output)[0].shape)
        input_shape_wo_batch_wo_pad = list(
            layer_backward(Input(self.output_dim_wo_batch))[0].shape
        )

        if layer.data_format == "channels_first":
            w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
            h_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]
        else:
            w_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]
            h_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]

        pad_layers = pooling_layer2D(w_pad, h_pad, layer.data_format)
        if len(pad_layers):
            layer_backward = Sequential([layer_backward] + pad_layers)
            # init
            layer_backward(Input(self.output_dim_wo_batch))
        self.layer_backward = layer_backward


class BackwardConv2DWithActivation(BackwardWithActivation):
    """
    This class implements a custom layer for backward pass of a `Conv2D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv2D
    from keras_custom.backward.layers import BackwardConv2DWithActivation

    # Assume `conv_layer` is a pre-defined Conv3D layer with an activation function
    backward_layer = BackwardConv2DWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """
    def __init__(
        self,
        layer: Conv2D,
        **kwargs,
    ):
        super().__init__(layer=layer, backward_linear=BackwardConv2D, backward_activation=BackwardActivation,**kwargs)


def get_backward_Conv2D(layer: Conv2D) -> Layer:
    """
    This function creates a `BackwardConv2D` layer based on a given `Conv2D` layer. It provides
    a convenient way to obtain the backward pass of the input `Conv2D` layer, using the
    `BackwardConv2D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `Conv2D` layer instance. The function uses this layer's configurations to set up the `BackwardConv2D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardConv2D`, which acts as the reverse layer for the given `Conv2D`.

    ### Example Usage:
    ```python
    from keras.layers import Conv2D
    from keras_custom.backward import get_backward_Conv2D

    # Assume `conv_layer` is a pre-defined Conv2D layer
    backward_layer = get_backward_Conv2D(conv_layer)
    output = backward_layer(input_tensor)
    """
    if layer.get_config()['activation']=='linear':
        return BackwardConv2D(layer)
    else:
        return BackwardConv2DWithActivation(layer)
