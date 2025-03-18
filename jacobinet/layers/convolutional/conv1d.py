import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.core.activations import BackwardActivation
from jacobinet.layers.layer import BackwardLinearLayer, BackwardWithActivation
from jacobinet.layers.utils import pooling_layer1D
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras.layers import Conv1D, Conv1DTranspose, Input  # type: ignore
from keras.models import Sequential  # type: ignore


def init_backward_conv1D(layer, input_dim_wo_batch, output_dim_wo_batch):
    dico_conv = layer.get_config()
    dico_conv.pop("groups")
    # input_shape = list(layer.input.shape[1:])
    input_shape = input_dim_wo_batch
    # update filters to match input, pay attention to data_format
    if layer.data_format == "channels_first":  # better to use enum than raw str
        dico_conv["filters"] = input_shape[0]
    else:
        dico_conv["filters"] = input_shape[-1]

    dico_conv["use_bias"] = False
    dico_conv["padding"] = "valid"

    layer_backward = Conv1DTranspose.from_config(dico_conv)
    layer_backward.kernel = layer.kernel

    layer_backward.built = True

    input_shape_wo_batch = input_dim_wo_batch
    input_shape_wo_batch_wo_pad = list(layer_backward(Input(output_dim_wo_batch))[0].shape)

    if layer.data_format == "channels_first":
        w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
    else:
        w_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]

    pad_layers = pooling_layer1D(w_pad, layer.data_format)
    if len(pad_layers):
        layer_backward = Sequential([layer_backward] + pad_layers)
        _ = layer_backward(Input(output_dim_wo_batch))

    return layer_backward


@keras.saving.register_keras_serializable()
class BackwardConv1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Conv1D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv1D
    from keras_custom.backward.layers import BackwardConv1D

    # Assume `conv_layer` is a pre-defined Conv1D layer
    backward_layer = BackwardConv1D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv1D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        self.layer_backward = init_backward_conv1D(
            layer, self.input_dim_wo_batch, self.output_dim_wo_batch
        )


class BackwardConv1DWithActivation(BackwardWithActivation):
    """
    This class implements a custom layer for backward pass of a `Conv1D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv1D
    from keras_custom.backward.layers import BackwardConv1DWithActivation

    # Assume `conv_layer` is a pre-defined Conv1D layer with an activation function
    backward_layer = BackwardConv1DWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv1D,
        **kwargs,
    ):
        super().__init__(
            layer=layer,
            backward_linear=BackwardConv1D,
            backward_activation=BackwardActivation,
            **kwargs,
        )


def get_backward_Conv1D(layer: Conv1D) -> Layer:
    """
    This function creates a `BackwardConv1D` layer based on a given `Conv1D` layer. It provides
    a convenient way to obtain the backward pass of the input `Conv1D` layer, using the
    `BackwardConv1D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `Conv1D` layer instance. The function uses this layer's configurations to set up the `BackwardConv1D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardConv1D`, which acts as the reverse layer for the given `Conv1D`.

    ### Example Usage:
    ```python
    from keras.layers import Conv1D
    from keras_custom.backward import get_backward_Conv1D

    # Assume `conv_layer` is a pre-defined Conv1D layer
    backward_layer = get_backward_Conv1D(conv_layer)
    output = backward_layer(input_tensor)
    """
    # check whether there is a non linear activation
    if layer.get_config()["activation"] == "linear":
        return BackwardConv1D(layer)
    else:
        return BackwardConv1DWithActivation(layer)
