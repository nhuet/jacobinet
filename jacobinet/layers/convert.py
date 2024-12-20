import logging
from typing import Any, Optional

from jacobinet.layers.layer import BackwardLinearLayer, BackwardLayer
from keras.layers import (
    Activation,
    Add,
    Average,
    Subtract,
    Concatenate,
    Maximum,
    Minimum,
    Multiply,
    Dense,
    EinsumDense,
    Layer,
    Conv2D,
    Conv1D,
    DepthwiseConv1D,
    DepthwiseConv2D,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Flatten,
    RepeatVector,
    Reshape,
    Permute,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    BatchNormalization,
    GroupNormalization,
    UnitNormalization,
    LayerNormalization,
    SpectralNormalization,
    LeakyReLU,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    MaxPooling2D,
    Dropout,
    ReLU,
    LeakyReLU,
    PReLU,
    ELU,
)

from jacobinet.layers import (
    get_backward_BatchNormalization,
    get_backward_Conv2D,
    get_backward_Conv1D,
    get_backward_DepthwiseConv1D,
    get_backward_DepthwiseConv2D,
    get_backward_ZeroPadding1D,
    get_backward_ZeroPadding2D,
    get_backward_ZeroPadding3D,
    get_backward_Cropping1D,
    get_backward_Cropping2D,
    get_backward_Cropping3D,
    get_backward_Flatten,
    get_backward_Reshape,
    get_backward_Permute,
    get_backward_RepeatVector,
    get_backward_AveragePooling2D,
    get_backward_AveragePooling1D,
    get_backward_AveragePooling3D,
    get_backward_GlobalAveragePooling2D,
    get_backward_GlobalAveragePooling1D,
    get_backward_GlobalAveragePooling3D,
    get_backward_MaxPooling2D,
    get_backward_Dense,
    get_backward_EinsumDense,
    get_backward_ReLU,
    get_backward_LeakyReLU,
    get_backward_PReLU,
    get_backward_ELU,
    get_backward_Activation,
    # merging
    get_backward_Add,
    get_backward_Average,
    get_backward_Subtract,
    get_backward_Concatenate,
    get_backward_Maximum,
    get_backward_Minimum,
    get_backward_Multiply,
)


# define BackwardLinear here to avoid circular import
def convert_to_backward(layer, use_bias):
    layer_backward = get_backward(layer, use_bias)
    return layer_backward


logger = logging.getLogger(__name__)

BACKWARD_PREFIX = "get_backward"

default_mapping_keras2backward_layer: dict[type[Layer], type[callable]] = {
    # convolution
    Conv2D: get_backward_Conv2D,
    Conv1D: get_backward_Conv1D,
    DepthwiseConv1D: get_backward_DepthwiseConv1D,
    DepthwiseConv2D: get_backward_DepthwiseConv2D,
    # reshaping
    ZeroPadding1D: get_backward_ZeroPadding1D,
    ZeroPadding2D: get_backward_ZeroPadding2D,
    ZeroPadding3D: get_backward_ZeroPadding3D,
    Cropping1D: get_backward_Cropping1D,
    Cropping2D: get_backward_Cropping2D,
    Cropping3D: get_backward_Cropping3D,
    Flatten: get_backward_Flatten,
    Reshape: get_backward_Reshape,
    Permute: get_backward_Permute,
    RepeatVector: get_backward_RepeatVector,
    # normalization
    BatchNormalization: get_backward_BatchNormalization,
    # pooling
    AveragePooling2D: get_backward_AveragePooling2D,
    AveragePooling1D: get_backward_AveragePooling1D,
    AveragePooling3D: get_backward_AveragePooling3D,
    GlobalAveragePooling2D: get_backward_GlobalAveragePooling2D,
    GlobalAveragePooling1D: get_backward_GlobalAveragePooling1D,
    GlobalAveragePooling3D: get_backward_GlobalAveragePooling3D,
    MaxPooling2D: get_backward_MaxPooling2D,
    # core
    Dense: get_backward_Dense,
    EinsumDense: get_backward_EinsumDense,
    # activations
    Activation: get_backward_Activation,
    ReLU: get_backward_ReLU,
    LeakyReLU: get_backward_LeakyReLU,
    PReLU: get_backward_PReLU,
    ELU: get_backward_ELU,
    # merging
    Add: get_backward_Add,
    Average: get_backward_Average,
    Subtract: get_backward_Subtract,
    Concatenate: get_backward_Concatenate,
    Maximum: get_backward_Maximum,
    Minimum: get_backward_Minimum,
    Multiply: get_backward_Multiply,
}
"""Default mapping between keras layers and get_backward callable"""


def get_backward(
    layer: Layer,
    mapping_keras2backward_classes: Optional[
        dict[type[Layer], type[BackwardLayer]]
    ] = None,
    **kwargs,
):
    keras_class = type(layer)
    """
    if isinstance(layer, BackwardLinearLayer):
        if use_bias:
            return layer.layer
        elif not hasattr(layer.layer, "use_bias"):
            return layer.layer
        else:
            # copy layer
            config = layer.layer.get_config()
            config["use_bias"] = False
            layer_ = layer.layer.__class__.from_config(config)
            layer_.weights = layer.layer.weights[:1]
            layer_.built = True
            return layer_
    """

    if mapping_keras2backward_classes is not None:
        default_mapping_keras2backward_layer.update(
            mapping_keras2backward_classes
        )

    get_backward_layer = default_mapping_keras2backward_layer.get(keras_class)
    if get_backward_layer is None:
        raise ValueError(
            "The backward mapping from the current class is not native and not available in mapping_keras2backward_layer, {} not found".format(
                keras_class
            )
        )
    return get_backward_layer(layer)
