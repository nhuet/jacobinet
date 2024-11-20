from keras.layers import Layer, PReLU
import keras.ops as K
from jacobinet.layers.layer import BackwardNonLinearLayer

from keras import KerasTensor as Tensor
from .prime import relu_prime


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
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        backward_output: Tensor = relu_prime(
            input,
            negative_slope=self.layer.alpha,
        )
        output = gradient * backward_output
        return output


def get_backward_PReLU(layer: PReLU, use_bias=True) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `PReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `PReLU` layer, using the
    `BackwardPReLU`.

    ### Parameters:
    - `layer`: A Keras `PReLU` layer instance. The function uses this layer's configurations to set up the `BackwardPReLU` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardPReLU`, which acts as the reverse layer for the given `PReLU`.

    ### Example Usage:
    ```python
    from keras.layers import PReLU
    from keras_custom.backward import get_backward_PReLU

    # Assume `activation_layer` is a pre-defined PReLU layer
    backward_layer = get_backward_PReLU(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardPReLU(layer, use_bias)
