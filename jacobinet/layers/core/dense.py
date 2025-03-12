from jacobinet.layers.layer import (
    BackwardLinearLayer,
    BackwardWithActivation,
)
from jacobinet.layers.core.activations import BackwardActivation
from keras.layers import Layer, Dense  # type: ignore
import keras.ops as K  # type: ignore

from keras import KerasTensor as Tensor


class BackwardDense(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Dense` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Dense
    from keras_custom.backward.layers import BackwardDense

    # Assume `dense_layer` is a pre-defined Dense layer
    backward_layer = BackwardDense(dense_layer)
    output = backward_layer(input_tensor)
    ```
    This code creates an instance of `BackwardDense` that will apply the transposed
    operation of `original_dense_layer` on the input tensor `inputs`.
    """

    def __init__(
        self,
        layer: Dense,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        return K.matmul(gradient, K.transpose(self.layer.kernel))


class BackwardDenseWithActivation(BackwardWithActivation):
    """ "
    This class implements a custom layer for backward pass of a `DepthwiseConv1D` layer in Keras with a non linear activation function.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv1
    from keras_custom.backward.layers import BackwardDenseWithActivation

    # Assume `dense_layer` is a pre-defined Dense layer with an activation function
    backward_layer = BackwardDenseWithActivation(dense_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Dense,
        **kwargs,
    ):
        super().__init__(
            layer=layer,
            backward_linear=BackwardDense,
            backward_activation=BackwardActivation,
            **kwargs,
        )


def get_backward_Dense(layer: Dense) -> Layer:
    """
    This function creates a `BackwardDense` layer based on a given `Dense` layer. It provides
    a convenient way to obtain the ackward pass of the input `Dense` layer, using the
    `BackwardDense` class.

    ### Parameters:
    - `layer`: A Keras `Dense` layer instance. The function uses this layer's configurations to set up the `BackwardDense` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardDense`, which acts as the reverse layer for the given `Dense`.

    ### Example Usage:
    ```python
    from keras.layers import Dense
    from keras_custom.backward import get_backward_Dense

    # Assume `dense_layer` is a pre-defined Dense layer
    backward_layer = get_backward_Dense(dense_layer)
    output = backward_layer(input_tensor)
    """
    if layer.get_config()["activation"] == "linear":
        return BackwardDense(layer)
    else:
        return BackwardDenseWithActivation(layer)
