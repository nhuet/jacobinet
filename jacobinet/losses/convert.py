from typing import Optional

from jacobinet.losses import BackwardLoss
from keras.losses import CategoricalCrossentropy  # type:ignore

from .categorical_crossentropy import get_backward_CategoricalCrossentropy
from .loss import CategoricalCrossentropy_Layer, Loss_Layer

default_mapping_keras2backward_loss: dict[type[Loss_Layer], type[callable]] = {
    # convolution
    CategoricalCrossentropy_Layer: get_backward_CategoricalCrossentropy,
}


def get_backward_loss(
    layer: Loss_Layer,
    mapping_keras2backward_losses: Optional[dict[type[Loss_Layer], type[BackwardLoss]]] = None,
    **kwargs,
) -> BackwardLoss:
    """
    Retrieves the corresponding backward loss layer for a given Keras loss layer.

    This function maps a Keras loss layer (e.g., `Loss_Layer`) to its corresponding
    backward loss layer (e.g., `BackwardLoss`) using an optional custom mapping.
    If no custom mapping is provided, a default mapping is used to retrieve the backward
    loss. If the loss layer type is not found in the mapping, a `ValueError` is raised.

    Args:
        layer: The Keras loss layer for which the backward loss layer
                             needs to be retrieved.
        mapping_keras2backward_losses:
            A dictionary mapping Keras loss layers to their corresponding backward loss layers.
            If `None`, the default mapping is used.
        **kwargs: Additional keyword arguments passed to the backward loss layer, if needed.

    Returns:
        BackwardLoss: The backward loss layer corresponding to the provided Keras loss layer.

    Raises:
        ValueError: If the backward loss layer cannot be found in the mapping for the provided
                    Keras loss layer type.

    Example:
        ```python
        backward_loss = get_backward_loss(my_loss_layer)
        ```
    """
    keras_class = type(layer)

    if mapping_keras2backward_losses is not None:
        default_mapping_keras2backward_loss.update(mapping_keras2backward_losses)

    get_backward_layer_loss = default_mapping_keras2backward_loss.get(keras_class)
    if get_backward_layer_loss is None:
        raise ValueError(
            "The backward mapping from the current loss is not native and not available in mapping_keras2backward_loss, {} not found".format(
                keras_class
            )
        )
    return get_backward_layer_loss(layer)
