from keras.layers import Layer, Average
from jacobinet.layers.merging import BackwardMergeLinearLayer

from keras import KerasTensor as Tensor


class BackwardAverage(BackwardMergeLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Average` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Average
    from keras_custom.backward.layers import BackwardAverage

    # Assume `mean_layer` is a pre-defined Average layer
    backward_layer = BackwardAverage(mean_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        coeff = 1.0 / self.n_input
        return [coeff * gradient] * self.n_input


def get_backward_Average(layer: Average, use_bias=True) -> Layer:
    """
    This function creates a `BackwardAverage` layer based on a given `Average` layer. It provides
    a convenient way to obtain the backward pass of the input `Average` layer, using the
    `BackwardAverage`.

    ### Parameters:
    - `layer`: A Keras `Average` layer instance. The function uses this layer's configurations to set up the `BackwardAverage` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAverage`, which acts as the reverse layer for the given `Average`.

    ### Example Usage:
    ```python
    from keras.layers import Average
    from keras_custom.backward import get_backward_Average

    # Assume `mean_layer` is a pre-defined Average layer
    backward_layer = get_backward_Average(mean_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardAverage(layer)
