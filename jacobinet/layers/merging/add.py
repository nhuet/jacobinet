from keras.layers import Layer, Add
from jacobinet.layers.merging import BackwardMergeLinearLayer

from keras import KerasTensor as Tensor


class BackwardAdd(BackwardMergeLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Add` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Add
    from keras_custom.backward.layers import BackwardAdd

    # Assume `add_layer` is a pre-defined Add layer
    backward_layer = BackwardAdd(add_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        return [gradient] * self.n_input


def get_backward_Add(layer: Add) -> Layer:
    """
    This function creates a `BackwardAdd` layer based on a given `Add` layer. It provides
    a convenient way to obtain the backward pass of the input `Add` layer, using the
    `BackwardAdd`.

    ### Parameters:
    - `layer`: A Keras `Add` layer instance. The function uses this layer's configurations to set up the `BackwardAdd` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAdd`, which acts as the reverse layer for the given `Add`.

    ### Example Usage:
    ```python
    from keras.layers import Add
    from keras_custom.backward import get_backward_Add

    # Assume `add_layer` is a pre-defined Add layer
    backward_layer = get_backward_Add(add_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAdd(layer)
