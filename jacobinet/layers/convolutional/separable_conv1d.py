from typing import Any, Optional

import keras
import numpy as np
from jacobinet.layers.convolutional.conv1d import BackwardConv1D
from jacobinet.layers.convolutional.depthwise_conv1d import BackwardDepthwiseConv1D
from jacobinet.layers.core.activations import BackwardActivation
from jacobinet.layers.layer import BackwardLinearLayer, BackwardWithActivation
from keras.layers import Conv1D, DepthwiseConv1D, Layer, SeparableConv1D  # type:ignore
from keras.models import Sequential  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardSeparableConv1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `SeparableConv1D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import SeparableConv1D
    from keras_custom.backward.layers import BackwardSeparableConv1D

    # Assume `conv_layer` is a pre-defined SeparableConv1D layer
    backward_layer = BackwardSeparableConv1D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: SeparableConv1D,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        # split into depthwise and conv
        layer_depthwise: DepthwiseConv1D
        layer_conv: Conv1D
        layer_depthwise, layer_conv = split_SeparableConv1D(
            self.layer, input_shape_wo_batch=self.input_dim_wo_batch
        )
        self.layer_backward_depthwise = BackwardDepthwiseConv1D(layer_depthwise)
        self.layer_backward_conv = BackwardConv1D(layer_conv)
        self.layer_backward = Sequential([self.layer_backward_conv, self.layer_backward_depthwise])


@keras.saving.register_keras_serializable()
class BackwardSeparableConv1DWithActivation(BackwardWithActivation):
    """
    This class implements a custom layer for backward pass of a `SeparableConv1D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import SeparableConv1D
    from keras_custom.backward.layers import BackwardSeparableConv2DWithActivation

    # Assume `conv_layer` is a pre-defined SeparableConv2D layer with an activation function
    backward_layer = BackwardSeparableConv2DWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv1D,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            backward_linear=BackwardSeparableConv1D,
            backward_activation=BackwardActivation,
            **kwargs,
        )


def split_SeparableConv1D(
    layer: SeparableConv1D, input_shape_wo_batch: Optional[tuple[int]] = None
) -> tuple[DepthwiseConv1D, Conv1D]:
    """
    split SeparableConv1D into a sequence of DepthwiseConv1D followed by Conv1D (pointwise).

    Args:
        layer: the SeparableConv1D layer

    Returns:
        tuple of depthwise_layer, conv_layer such that conv_layer(depthwise_layer) = layer
        Notes: layers have shared weights with the initial layer
    """
    # filters, kernel_size, strides=(1, 1), padding='same', use_bias=False, depth_multiplier=1, name=None
    filters = layer.filters
    kernel_size = layer.kernel_size[0]
    strides = layer.strides
    padding = layer.padding
    use_bias = layer.use_bias
    depth_multiplier = layer.depth_multiplier
    name = layer.name

    layer_depthwise = DepthwiseConv1D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        use_bias=False,
        name=None if not name else name + "_depthwise",
    )
    layer_conv = Conv1D(
        filters,
        kernel_size=(1,),
        padding="same",
        use_bias=use_bias,
        name=None if not name else name + "_pointwise",
    )

    layer_depthwise.kernel = layer.depthwise_kernel
    layer_depthwise.built = True
    layer_conv._kernel = layer.pointwise_kernel
    layer_conv.bias = layer.bias

    layer_conv.built = True
    model = Sequential([layer_depthwise, layer_conv])
    try:
        model(layer.input)
    except:
        if input_shape_wo_batch is None:
            raise ValueError("input_shape_wo_batch cannot be None")
        input_np = np.zeros((1,) + input_shape_wo_batch)
        model(input_np)
    return layer_depthwise, layer_conv


def get_backward_SeparableConv1D(layer: Conv1D) -> Layer:
    """
    This function creates a `BackwardSeparableConv1D` layer based on a given `SeparableConv1D` layer. It provides
    a convenient way to obtain the backward pass of the input `SeparableConv1D` layer, using the
    `BackwardSeparableConv1D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `SeparableConv1D` layer instance. The function uses this layer's configurations to set up the `BackwardSeparableConv1D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardSeparableConv1D`, which acts as the reverse layer for the given `SeparableConv1D`.

    ### Example Usage:
    ```python
    from keras.layers import SeparableConv1D
    from keras_custom.backward import get_backward_SeparableConv1D

    # Assume `conv_layer` is a pre-defined SeparableConv1D layer
    backward_layer = get_backward_SeparableConv1D(conv_layer)
    output = backward_layer(input_tensor)
    """
    if layer.get_config()["activation"] == "linear":
        return BackwardSeparableConv1D(layer)
    else:
        return BackwardSeparableConv1DWithActivation(layer)
