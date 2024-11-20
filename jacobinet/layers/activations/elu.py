from keras.layers import Layer, ELU
import keras.ops as K
from jacobinet.layers.layer import BackwardNonLinearLayer

from keras import KerasTensor as Tensor
from .prime import elu_prime


class BackwardELU(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `ELU` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import ELU
    from keras_custom.backward.layers import BackwardELU

    # Assume `activation_layer` is a pre-defined ELU layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: ELU,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        backward_output: Tensor = elu_prime(input, alpha=self.layer.alpha)
        output = gradient * backward_output
        return output


def get_backward_ELU(layer: ELU, use_bias=True) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `ELU` layer. It provides
    a convenient way to obtain the backward pass of the input `ELU` layer, using the
    `BackwardELU`.

    ### Parameters:
    - `layer`: A Keras `ELU` layer instance. The function uses this layer's configurations to set up the `BackwardELU` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardELU`, which acts as the reverse layer for the given `ELU`.

    ### Example Usage:
    ```python
    from keras.layers import ELU
    from keras_custom.backward import get_backward_ELU

    # Assume `activation_layer` is a pre-defined ELU layer
    backward_layer = get_backward_ELU(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardELU(layer, use_bias)
