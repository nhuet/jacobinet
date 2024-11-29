from keras.layers import Layer, Activation
import keras.ops as K
from jacobinet.layers.layer import BackwardNonLinearLayer

from keras import KerasTensor as Tensor
from jacobinet.layers.activations.prime import deserialize


class BackwardActivation(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Activation` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Activation
    from keras_custom.backward.layers import BackwardActivation

    # Assume `activation_layer` is a pre-defined Activation layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Activation,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        activation_name = layer.get_config()["activation"]
        self.layer_backward = deserialize(activation_name)

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        backward_output: Tensor = self.layer_backward(input)
        output = gradient * backward_output
        return output


def get_backward_Activation(layer: Activation) -> Layer:
    """
    This function creates a `BackwardActivation` layer based on a given `Activation` layer. It provides
    a convenient way to obtain the backward pass of the input `Activation` layer, using the
    `BackwardActivation`.

    ### Parameters:
    - `layer`: A Keras `Activation` layer instance. The function uses this layer's configurations to set up the `BackwardActivation` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardActivation`, which acts as the reverse layer for the given `Activation`.

    ### Example Usage:
    ```python
    from keras.layers import Activation
    from keras_custom.backward import get_backward_Activation

    # Assume `activation_layer` is a pre-defined Activation layer
    backward_layer = get_backward_Activation(activation_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardActivation(layer)
