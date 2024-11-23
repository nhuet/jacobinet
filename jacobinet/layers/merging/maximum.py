from keras.layers import Layer, Maximum
import keras.ops as K
from jacobinet.layers.merging import BackwardMergeNonLinearLayer

from keras import KerasTensor as Tensor


class BackwardMaximum(BackwardMergeNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Maximum` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Maximum
    from keras_custom.backward.layers import BackwardMaximum

    # Assume `maximum_layer` is a pre-defined Maximum layer
    backward_layer = BackwardMaximum(maximum_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):

        output_layer = self.layer(input)
        return [gradient*(K.sign(input_i - output_layer)+1)for input_i in input]


def get_backward_Maximum(layer: Maximum, use_bias=True) -> Layer:
    """
    This function creates a `BackwardMaximum` layer based on a given `Maximum` layer. It provides
    a convenient way to obtain the backward pass of the input `Maximum` layer, using the
    `BackwardMaximum`.

    ### Parameters:
    - `layer`: A Keras `Maximum` layer instance. The function uses this layer's configurations to set up the `BackwardMaximum` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMaximum`, which acts as the reverse layer for the given `Maximum`.

    ### Example Usage:
    ```python
    from keras.layers import Maximum
    from keras_custom.backward import get_backward_Maximum

    # Assume `maximum_layer` is a pre-defined Maximum layer
    backward_layer = get_backward_Maximum(concat_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardMaximum(layer)
