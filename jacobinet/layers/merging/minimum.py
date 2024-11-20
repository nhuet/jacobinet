from keras.layers import Layer, Minimum
import keras.ops as K
from jacobinet.layers.merging import BackwardMergeNonLinearLayer

from keras import KerasTensor as Tensor


class BackwardMinimum(BackwardMergeNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Minimum` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Minimum
    from keras_custom.backward.layers import BackwardMinimum

    # Assume `minimum_layer` is a pre-defined Minimum layer
    backward_layer = BackwardMinimum(minimum_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        output_layer = self.layer(input)
        return [
            gradient * (2 * K.sign(output_layer - input_i) + 1)
            for input_i in input
        ]

    """
    def call(self, inputs, training=None, mask=None):
        layer_output = inputs[0]
        layer_inputs = inputs[1:]

        reshape_tag, layer_output, n_out = reshape_to_batch(inputs, list(self.layer.output.shape))
        output_layer = self.layer(layer_inputs)
        output = [ layer_output*(2*K.sign(output_layer - input_i)+1) for input_i in layer_inputs]
        
        if reshape_tag:
            output = [K.reshape(output[i], [-1]+n_out+list(self.input_dim_wo_batch[i])) for i in range(self.n_input)]   
                
        return output
    """


def get_backward_Minimum(layer: Minimum, use_bias=True) -> Layer:
    """
    This function creates a `BackwardMinimum` layer based on a given `Minimum` layer. It provides
    a convenient way to obtain the backward pass of the input `Minimum` layer, using the
    `BackwardMinimum`.

    ### Parameters:
    - `layer`: A Keras `Minimum` layer instance. The function uses this layer's configurations to set up the `BackwardMinimum` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMinimum`, which acts as the reverse layer for the given `Minimum`.

    ### Example Usage:
    ```python
    from keras.layers import Minimum
    from keras_custom.backward import get_backward_Minimum

    # Assume `minimum_layer` is a pre-defined Minimum layer
    backward_layer = get_backward_Minimum(minimum_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardMinimum(layer)
