import numpy as np
import keras  # type:ignore
from keras.layers import Layer  # type:ignore
import numpy as np
from keras.losses import Loss, CategoricalCrossentropy  # type:ignore


CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
# MeanSquaredError
# MeanAbsoluteError
# MeanAbsoluteError
# MeanSquaredLogarithmicError
# CosineSimilarity
# LogCosh
# Hinge
# SquaredHinge
# CategoricalHinge
# KLDivergence
# BinaryCrossentropy

import keras

from typing import Optional


def deserialize(name: str) -> Loss:
    """Get the activation from name.

    Args:
        name: name of the method.
    among the implemented Keras activation function.

    Returns:
        the activation function
    """
    name = name.lower()
    if name == CATEGORICAL_CROSSENTROPY:
        return keras.losses.CategoricalCrossentropy()
    raise ValueError("unknown loss")


class Loss_Layer(Layer):
    """
    A custom Keras layer that computes a loss function between true labels and predicted values.

    This layer wraps a given loss function (e.g., `Loss`) and computes the loss during
    the forward pass by comparing the true labels (`y_true`) with the predicted values (`y_pred`).
    The loss is calculated in a flattened format (excluding the batch dimension) and returned.

    Args:
        loss: A callable loss function that computes the loss between the true and predicted values.
        **kwargs: Additional keyword arguments passed to the parent class (`Layer`).


    Example:
        ```python
        loss_layer = Loss_Layer(loss_function)
        loss_value = loss_layer(inputs)
        ```
    """

    def __init__(self, loss: Loss, **kwargs):
        super(Loss_Layer, self).__init__(**kwargs)
        self.loss = loss

    def call(self, inputs_):

        y_true, y_pred = inputs_
        # reshape to have batch dimension
        shape_wo_batch = np.prod(y_true.shape[1:])
        y_true_flatten = keras.ops.reshape(y_true, [-1, shape_wo_batch])
        loss = self.loss(y_true, y_pred)

        return keras.ops.zeros_like(y_true_flatten[:, :1]) + loss

    # serialization to do


class CategoricalCrossentropy_Layer(Loss_Layer):
    """
    A custom Keras layer for computing categorical cross-entropy loss between true labels and predictions.

    This class wraps the `CategoricalCrossentropy` loss function and computes the loss during
    the forward pass by comparing the true labels (`y_true`) and predicted values (`y_pred`).
    It inherits from `Loss_Layer` and is specifically designed for the categorical cross-entropy loss function.

    Args:
        loss (CategoricalCrossentropy): A `CategoricalCrossentropy` loss function that computes
                                        the loss between the true labels and predicted values.
        **kwargs: Additional keyword arguments passed to the parent class (`Loss_Layer`).


    Example:
        ```python
        categorical_crossentropy_layer = CategoricalCrossentropy_Layer(loss=categorical_crossentropy_instance)
        loss_value = categorical_crossentropy_layer([y_true, y_pred])
        ```

    """

    loss: CategoricalCrossentropy


default_mapping_loss2layer: dict[type[Loss], type[Loss_Layer]] = {
    # convolution
    CategoricalCrossentropy: CategoricalCrossentropy_Layer,
}


def get_loss_as_layer(
    loss: Loss,
    mapping_loss2layer: Optional[dict[type[Loss], type[Loss_Layer]]] = None,
    **kwargs,
) -> Loss_Layer:
    """
    Retrieves the corresponding loss layer for a given Keras loss function.

    This function maps a Keras loss function (e.g., `Loss`) to its corresponding custom
    loss layer (e.g., `Loss_Layer`). If no custom mapping is provided, the function uses
    a default mapping to retrieve the loss layer. If the loss function type is not found
    in the mapping, a `ValueError` is raised.

    Args:
        loss: A Keras loss function to be converted into a corresponding loss layer.
        mapping_loss2layer: A dictionary that maps
            Keras loss functions to their corresponding loss layers. If `None`, the default mapping
            (`default_mapping_loss2layer`) will be used.
        **kwargs: Additional keyword arguments passed to the loss layer constructor, if needed.

    Returns:
        Loss_Layer: The corresponding loss layer for the provided Keras loss function.

    Raises:
        ValueError: If the loss function cannot be mapped to a loss layer, a `ValueError` is raised.

    Example:
        ```python
        loss_layer = get_loss_as_layer(categorical_crossentropy_loss)
        ```
    """

    keras_class = type(loss)

    if mapping_loss2layer is not None:
        default_mapping_loss2layer.update(mapping_loss2layer)

    get_loss_as_layer = default_mapping_loss2layer.get(keras_class)
    if get_loss_as_layer is None:
        raise ValueError(
            "The mapping from the current loss to a layer is not native and not available in mapping_loss2layer, {} not found".format(
                keras_class
            )
        )

    return get_loss_as_layer(loss)
