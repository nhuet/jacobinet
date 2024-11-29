from keras.layers import BatchNormalization, Layer
from jacobinet.layers.layer import BackwardLinearLayer
from keras.src import backend
from keras.src import ops
import keras


class BackwardBatchNormalization(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `BatchNormalization` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the normalization
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import BatchNormalization
    from keras_custom.backward.layers import BackwardBatchNormalization

    # Assume `batch_norm_layer` is a pre-defined BatchNormalization layer
    backward_layer = BackwardBatchNormalization(batch_norm_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self, gradient, input=None, training=None, mask=None
    ):
        if mask is not None:
            if len(mask.shape) != len(gradient.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={gradient.shape}, inputs.shape={gradient.shape}"
                )

        compute_dtype = backend.result_type(gradient.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        gradient = ops.cast(gradient, compute_dtype)

        moving_mean = ops.cast(self.layer.moving_mean, gradient.dtype)
        moving_variance = ops.cast(self.layer.moving_variance, gradient.dtype)

        if training and self.layer.trainable:
            mean, variance = self.layer._moments(gradient, mask)
        else:
            mean = moving_mean
            variance = moving_variance

        if self.layer.scale:
            gamma = ops.cast(self.layer.gamma, gradient.dtype)
        else:
            gamma = None

        if self.layer.center:
            beta = ops.cast(self.layer.beta, gradient.dtype)
        else:
            beta = None

        # reshape mean, variance, beta, gamme to the right shape
        input_dim_batch = [-1] + [1] * (len(gradient.shape) - 1)
        input_dim_batch[self.layer.axis] = gradient.shape[self.layer.axis]

        mean_ = ops.reshape(mean, input_dim_batch)
        variance_ = ops.reshape(variance, input_dim_batch)
        gamma_ = ops.reshape(gamma, input_dim_batch)
        beta_ = ops.reshape(beta, input_dim_batch)

        # z = (x-mean_)/ops.sqrt(variance_+epsilon)
        # inputs = gamma_*z + beta_
        # thus z = (inputs-beta_)/gamma_
        # thus x = z*ops.sqrt(variance_+epsilon) + mean_
        # z = (inputs -beta_)/gamma_
        # x = z*ops.sqrt(variance_+self.layer.epsilon) + mean_
        w = ops.sqrt(variance_ + self.layer.epsilon) / gamma_
        outputs = w * gradient

        return outputs

    """
    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        reshape_tag, inputs, n_out = reshape_to_batch(inputs, list(self.layer.output.shape))
        
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        compute_dtype = backend.result_type(inputs.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        inputs = ops.cast(inputs, compute_dtype)

        moving_mean = ops.cast(self.layer.moving_mean, inputs.dtype)
        moving_variance = ops.cast(self.layer.moving_variance, inputs.dtype)

        if training and self.layer.trainable:
            mean, variance = self.layer._moments(inputs, mask)
        else:
            mean = moving_mean
            variance = moving_variance

        if self.layer.scale:
            gamma = ops.cast(self.layer.gamma, inputs.dtype)
        else:
            gamma = None

        if self.layer.center:
            beta = ops.cast(self.layer.beta, inputs.dtype)
        else:
            beta = None

        # reshape mean, variance, beta, gamme to the right shape
        input_dim_batch = [-1] + [1] * (len(inputs.shape) - 1)
        input_dim_batch[self.layer.axis] = inputs.shape[self.layer.axis]

        mean_ = ops.reshape(mean, input_dim_batch)
        variance_ = ops.reshape(variance, input_dim_batch)
        gamma_ = ops.reshape(gamma, input_dim_batch)
        beta_ = ops.reshape(beta, input_dim_batch)

        # z = (x-mean_)/ops.sqrt(variance_+epsilon)
        # inputs = gamma_*z + beta_
        # thus z = (inputs-beta_)/gamma_
        # thus x = z*ops.sqrt(variance_+epsilon) + mean_
        # z = (inputs -beta_)/gamma_
        # x = z*ops.sqrt(variance_+self.layer.epsilon) + mean_
        w = ops.sqrt(variance_ + self.layer.epsilon) / gamma_
        outputs = w * inputs

        if self.use_bias:
            b = -beta_ * ops.sqrt(variance_ + self.layer.epsilon) / gamma_ + mean_
            return outputs + b

        if reshape_tag:
            outputs = keras.ops.reshape(outputs, [-1]+n_out+list(self.layer.input.shape[1:]))

        return outputs
    """


def get_backward_BatchNormalization(
    layer: BatchNormalization
) -> Layer:
    """
    This function creates a `BackwardBatchNormalization` layer based on a given `BatchNormalization` layer. It provides
    a convenient way to obtain a backward approximation of the input `BatchNormalization` layer, using the
    `BackwardBatchNormalization` class to reverse the batch normalization operation.

    ### Parameters:
    - `layer`: A Keras `BatchNormalization` layer instance. The function uses this layer's configurations to set up the `BackwardBatchNormalization` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardBatchNormalization`, which acts as the reverse layer for the given `BatchNormalization`.

    ### Example Usage:
    ```python
    from keras.layers import BatchNormalization
    from my_custom_layers import get_backward_BatchNormalization

    # Assume `batch_norm_layer` is a pre-defined BatchNormalization layer
    backward_layer = get_backward_BatchNormalization(batch_norm_layer)
    output = backward_layer(input_tensor)
    """
    layer_backward = BackwardBatchNormalization(layer=layer)

    return layer_backward
