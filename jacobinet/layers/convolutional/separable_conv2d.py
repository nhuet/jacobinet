from typing import Any, Optional

import keras
import numpy as np
from jacobinet.layers.convolutional.conv2d import BackwardConv2D
from jacobinet.layers.convolutional.depthwise_conv2d import BackwardDepthwiseConv2D
from jacobinet.layers.core.activations import BackwardActivation
from jacobinet.layers.layer import BackwardLinearLayer, BackwardWithActivation
from keras.layers import Conv2D, DepthwiseConv2D, Layer, SeparableConv2D  # type:ignore
from keras.models import Sequential  # type:ignore


@keras.saving.register_keras_serializable()
class BackwardSeparableConv2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `SeparableConv2D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import SeparableConv2D
    from keras_custom.backward.layers import BackwardSeparableConv2D

    # Assume `conv_layer` is a pre-defined SeparableConv2D layer
    backward_layer = BackwardSeparableConv2D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: SeparableConv2D,
        **kwargs: Any,
    ):
        super().__init__(layer=layer, **kwargs)
        # split into depthwise and conv
        layer_depthwise: DepthwiseConv2D
        layer_conv: Conv2D
        layer_depthwise, layer_conv = split_SeparableConv2D(
            self.layer, input_shape_wo_batch=self.input_dim_wo_batch
        )
        self.layer_backward_depthwise = BackwardDepthwiseConv2D(layer_depthwise)
        self.layer_backward_conv = BackwardConv2D(layer_conv)
        self.layer_backward = Sequential([self.layer_backward_conv, self.layer_backward_depthwise])


@keras.saving.register_keras_serializable()
class BackwardSeparableConv2DWithActivation(BackwardWithActivation):
    """
    This class implements a custom layer for backward pass of a `SeparableConv2D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import SeparableConv2D
    from keras_custom.backward.layers import BackwardSeparableConv2DWithActivation

    # Assume `conv_layer` is a pre-defined SeparableConv2D layer with an activation function
    backward_layer = BackwardSeparableConv2DWithActivation(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv2D,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            backward_linear=BackwardSeparableConv2D,
            backward_activation=BackwardActivation,
            **kwargs,
        )


def split_SeparableConv2D(
    layer: SeparableConv2D, input_shape_wo_batch: Optional[tuple[int]] = None
) -> tuple[DepthwiseConv2D, Conv2D]:
    """
    split SeparableConv2D into a sequence of DepthwiseConv2D followed by Conv2D (pointwise).

    Args:
        layer: the SeparableConv2D layer

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

    layer_depthwise = DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        use_bias=False,
        name=None if not name else name + "_depthwise",
    )
    layer_conv = Conv2D(
        filters,
        kernel_size=(1, 1),
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


def get_backward_SeparableConv2D(layer: Conv2D) -> Layer:
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
    if layer.get_config()["activation"] == "linear":
        return BackwardSeparableConv2D(layer)
    else:
        return BackwardSeparableConv2DWithActivation(layer)
