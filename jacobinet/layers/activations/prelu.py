import keras
from keras.layers import Layer, PReLU  # type: ignore
import keras.ops as K  # type: ignore
from keras import KerasTensor as Tensor

from jacobinet.layers.layer import BackwardNonLinearLayer

from .prime import relu_prime

@keras.saving.register_keras_serializable()
class BackwardPReLU(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `PReLU` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import PReLU
    from keras_custom.backward.layers import BackwardPReLU

    # Assume `activation_layer` is a pre-defined PReLU layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: PReLU,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        backward_output: Tensor = relu_prime(
            input,
            negative_slope=self.layer.alpha,
        )
        output = gradient * backward_output
        return output


def get_backward_PReLU(layer: PReLU) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `PReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `PReLU` layer, using the
    `BackwardPReLU`.

    ### Parameters:
    - `layer`: A Keras `PReLU` layer instance. 
    The function uses this layer's configurations to set up the `BackwardPReLU` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardPReLU`, 
    which acts as the reverse layer for the given `PReLU`.

    ### Example Usage:
    ```python
    from keras.layers import PReLU
    from keras_custom.backward import get_backward_PReLU

    # Assume `activation_layer` is a pre-defined PReLU layer
    backward_layer = get_backward_PReLU(activation_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardPReLU(layer)
