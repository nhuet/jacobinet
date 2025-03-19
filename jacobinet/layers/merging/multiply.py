import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.merging.base_merge import BackwardMergeNonLinearLayer
from keras.layers import Layer, Multiply  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardMultiply(BackwardMergeNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Multiply` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Multiply
    from keras_custom.backward.layers import BackwardMultiply

    # Assume `multiply_layer` is a pre-defined Muliply layer
    backward_layer = BackwardMultiply(multiply_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        layer_input_0 = input[0]
        layer_input_1 = input[1]

        return [gradient * layer_input_1, gradient * layer_input_0]

    """
    def call(self, inputs, training=None, mask=None):
        layer_output = inputs[0]
        layer_input_0 = inputs[1]
        layer_input_1 = inputs[2]

        reshape_tag, layer_output, n_out = reshape_to_batch(inputs, list(self.layer.output.shape))

        output = [layer_output*layer_input_1, layer_output*layer_input_0]

        if reshape_tag:
            output = [K.reshape(output[i], [-1]+n_out+list(self.input_dim_wo_batch[i])) for i in range(self.n_input)]

        return output
    """


def get_backward_Multiply(layer: Multiply) -> Layer:
    """
    This function creates a `BackwardMultiply` layer based on a given `Multiply` layer. It provides
    a convenient way to obtain the backward pass of the input `Multiply` layer, using the
    `BackwardMultiply`.

    ### Parameters:
    - `layer`: A Keras `Multiply` layer instance. The function uses this layer's configurations to set up the `BackwardMultiply` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMultiply`, which acts as the reverse layer for the given `Multiply`.

    ### Example Usage:
    ```python
    from keras.layers import Multiply
    from keras_custom.backward import get_backward_Multiply

    # Assume `multiply_layer` is a pre-defined Multiply layer
    backward_layer = get_backward_Multiply(concat_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardMultiply(layer)
