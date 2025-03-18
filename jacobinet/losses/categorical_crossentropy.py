import keras
import keras.ops as K  # type:ignore
from .base_loss import BackwardLoss
from .loss import CategoricalCrossentropy_Layer

@keras.saving.register_keras_serializable()
class BackwardCrossentropy(BackwardLoss):
    """
    A custom backward loss layer for computing gradients in cross-entropy loss functions,
    specifically for categorical cross-entropy with a softmax activation.

    This class is designed to compute the backward gradients for the categorical cross-entropy
    loss function, where the forward pass typically involves softmax. The gradients are computed
    with respect to both the true labels (`y_true`) and the predicted logits (`y_pred`).

    Args:
        layer: The layer that provides the loss function used for
                                                cross-entropy and its associated parameters.

    Raises:
        NotImplementedError: If the `loss.from_logits` is `False` and the softmax as an activation
                              has not been implemented yet.

    Example:
        ```python
        backward_crossentropy = BackwardCrossentropy(layer=my_loss_layer)
        gradients = backward_crossentropy.call_on_reshaped_gradient(gradient, input_data)
        ```
    """

    layer: CategoricalCrossentropy_Layer

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        if self.layer.loss.from_logits:
            y_true, y_pred = input
            n_class = y_pred.shape[self.layer.loss.axis]
            softmax = K.softmax(y_pred, axis=self.layer.loss.axis)
            gradient_ = K.eye(n_class)[None] * K.expand_dims(
                (1 - softmax), self.layer.loss.axis
            )

            grad_crossentropy = K.sum(
                K.expand_dims(y_true, self.layer.loss.axis) * gradient_,
                self.layer.loss.axis,
            )  # (None, n_class)

            grad_y_true = gradient * K.log(softmax)
            grad_y_pred = gradient * grad_crossentropy

            return [grad_y_true, grad_y_pred]
        else:
            raise NotImplementedError(
                "The softmax activation for categorical cross-entropy with from_logits=False is not implemented. "
                "If you need this functionality, please raise a dedicated pull request to implement it."
            )


def get_backward_CategoricalCrossentropy(
    layer: CategoricalCrossentropy_Layer,
) -> BackwardCrossentropy:
    """
    This function creates a `BackwardCrossentropy` layer based on a given `CategoricalCrossentropy_Layer` layer. It provides
    a convenient way to obtain a backward approximation of the input `CategoricalCrossentropy_Layer` layer, using the
    `BackwardCrossentropy` class to reverse the flatten operation.

    ### Parameters:
    - `layer`: A Keras `CategoricalCrossentropy_Layer` layer instance. The function uses this layer's configurations to set up the `BackwardCrossentropy` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardCrossentropy`, which acts as the reverse layer for the given `CategoricalCrossentropy_Layer`.

    ### Example Usage:
    ```python
    from keras.losses import CategoricalCrossentropy
    from jacobinet.losses get_loss_as_layer
    from keras_custom.backward import get_backward_Flatten

    # Assume `loss` is a pre-defined CategoricalCrossentropy loss
    loss_layer = get_loss_as_layer(loss)
    backward_layer = get_backward_Flatten(loss_layer)
    output = backward_layer([y_true, y_pred])
    """

    return BackwardCrossentropy(layer)
