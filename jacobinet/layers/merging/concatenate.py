from keras.layers import Layer, Concatenate
import keras.ops as K

from jacobinet.layers.merging import BackwardMergeLinearLayer

from keras import KerasTensor as Tensor


class BackwardConcatenate(BackwardMergeLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Concatenate` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Concatenate
    from keras_custom.backward.layers import BackwardConcatenate

    # Assume `concat_layer` is a pre-defined Concatenate layer
    backward_layer = BackwardConcatenate(concat_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Concatenate,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        self.dims = [
            ([1] + layer_i_shape)[self.layer.axis]
            for layer_i_shape in self.input_dim_wo_batch
        ]

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        return K.split(gradient, indices_or_sections=K.cumsum(self.dims[:-1]), axis=self.layer.axis)


def get_backward_Concatenate(layer: Concatenate, use_bias=True) -> Layer:
    """
    This function creates a `BackwardConcatenate` layer based on a given `Concatenate` layer. It provides
    a convenient way to obtain the backward pass of the input `Concatenate` layer, using the
    `BackwardConcatenate`.

    ### Parameters:
    - `layer`: A Keras `Concatenate` layer instance. The function uses this layer's configurations to set up the `BackwardConcatenate` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardConcatenate`, which acts as the reverse layer for the given `Concatenate`.

    ### Example Usage:
    ```python
    from keras.layers import Concatenate
    from keras_custom.backward import get_backward_Concatenate

    # Assume `concat_layer` is a pre-defined Concatenate layer
    backward_layer = get_backward_Concatenate(concat_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardConcatenate(layer)
