import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardNonLinearLayer
from keras import KerasTensor as Tensor
from keras.layers import Layer, LeakyReLU  # type: ignore

from .prime import leaky_relu_prime


@keras.saving.register_keras_serializable()
class BackwardLeakyReLU(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `LeakyReLU` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import LeakyReLU
    from keras_custom.backward.layers import BackwardLeakyReLU

    # Assume `activation_layer` is a pre-defined LeakyReLU layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: LeakyReLU,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        backward_output: Tensor = leaky_relu_prime(input, negative_slope=self.layer.negative_slope)
        output = gradient * backward_output
        return output


def get_backward_LeakyReLU(layer: LeakyReLU) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `LeakyReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `LeakyReLU` layer, using the
    `BackwardLeakyReLU`.

    ### Parameters:
    - `layer`: A Keras `LeakyReLU` layer instance. The function uses this layer's configurations to set up the `BackwardLeakyReLU` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLeakyReLU`, which acts as the reverse layer for the given `LeakyReLU`.

    ### Example Usage:
    ```python
    from keras.layers import LeakyReLU
    from keras_custom.backward import get_backward_LeakyReLU

    # Assume `activation_layer` is a pre-defined LeakyReLU layer
    backward_layer = get_backward_LeakyReLU(activation_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardLeakyReLU(layer)
