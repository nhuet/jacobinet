from keras.layers import Layer, Dot
import keras.ops as K
from jacobinet.layers.merging import BackwardMergeNonLinearLayer

from keras import KerasTensor as Tensor


class BackwardDot(BackwardMergeNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Dot` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Dot
    from keras_custom.backward.layers import BackwardDot

    # Assume `dot_layer` is a pre-defined Dot layer
    backward_layer = BackwardDot(dot_layer)
    output = backward_layer(input_tensor)
    """

    def call(self, inputs, training=None, mask=None):
        layer_output = inputs[0]
        layer_input_0 = inputs[1]
        layer_input_1 = inputs[2]
        axis_0 = self.layer.axis[0]
        axis_1 = self.layer.axis[1]

        reshape_tag, layer_output, n_out = reshape_to_batch(
            inputs, list(self.layer.output.shape)
        )

        raise NotImplementedError()
        output = [layer_output * layer_input_1, layer_output * layer_input_0]

        if reshape_tag:
            output = [
                K.reshape(
                    output[i], [-1] + n_out + list(self.input_dim_wo_batch[i])
                )
                for i in range(self.n_input)
            ]

        return output


def get_backward_Dot(layer: Dot) -> Layer:
    """
    This function creates a `BackwardDot` layer based on a given `Dot` layer. It provides
    a convenient way to obtain the backward pass of the input `Dot` layer, using the
    `BackwardDot`.

    ### Parameters:
    - `layer`: A Keras `Dot` layer instance. The function uses this layer's configurations to set up the `BackwardDot` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardDot`, which acts as the reverse layer for the given `Dot`.

    ### Example Usage:
    ```python
    from keras.layers import Dot
    from keras_custom.backward import get_backward_Dot

    # Assume `dot_layer` is a pre-defined Dot layer
    backward_layer = get_backward_Dot(dot_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardDot(layer)
