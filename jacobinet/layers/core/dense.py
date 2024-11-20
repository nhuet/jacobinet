from jacobinet.layers.layer import BackwardLinearLayer
from keras.layers import Layer, Dense
import keras.ops as K


class BackwardDense(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Dense` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Dense
    from keras_custom.backward.layers import BackwardDense

    # Assume `dense_layer` is a pre-defined Dense layer
    backward_layer = BackwardDense(dense_layer)
    output = backward_layer(input_tensor)
    ```
    This code creates an instance of `BackwardDense` that will apply the transposed
    operation of `original_dense_layer` on the input tensor `inputs`.
    """

    def __init__(
        self,
        layer: Dense,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        if self.layer.use_bias and self.use_bias:
            gradient = gradient - self.layer(
                K.zeros([1] + self.output_dim_wo_batch)
            )

        if self.layer.use_bias and self.use_bias:
            input_dim_wo_batch = list(self.output_dim_wo_batch)
            output = K.add(
                gradient, -self.layer(K.zeros([1] + input_dim_wo_batch))
            )
            output = K.matmul(output, K.transpose(self.layer.kernel))
        else:
            output = K.matmul(gradient, K.transpose(self.layer.kernel))

        return output


def get_backward_Dense(layer: Dense, use_bias=True) -> Layer:
    """
    This function creates a `BackwardDense` layer based on a given `Dense` layer. It provides
    a convenient way to obtain the ackward pass of the input `Dense` layer, using the
    `BackwardDense` class.

    ### Parameters:
    - `layer`: A Keras `Dense` layer instance. The function uses this layer's configurations to set up the `BackwardDense` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardDense`, which acts as the reverse layer for the given `Dense`.

    ### Example Usage:
    ```python
    from keras.layers import Dense
    from keras_custom.backward import get_backward_Dense

    # Assume `dense_layer` is a pre-defined Dense layer
    backward_layer = get_backward_Dense(dense_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardDense(layer, use_bias)
