from keras.layers import ZeroPadding1D, Cropping1D, Cropping2D, Reshape, Input  # type: ignore
from keras.layers import Layer  # type: ignore
from keras.models import Sequential  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer


class BackwardZeroPadding1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `ZeroPadding1D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the zero padding
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import ZeroPadding1D
    from keras_custom.backward.layers import BackwardZeroPadding1D

    # Assume `cropping_layer` is a pre-defined ZeroPadding1D layer
    backward_layer = BackwardZeroPadding1D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: ZeroPadding1D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_padding = layer.get_config()
        padding = dico_padding["padding"]
        data_format = dico_padding["data_format"]
        if data_format == "channels_last":
            self.layer_backward = Cropping1D(cropping=padding)
            self.layer_backward.built = True
        else:
            # Cropping1D is only working on axis=1, we need to use Cropping2D instead with 0 padding along axis=1
            layer_reshape_b = Reshape(
                target_shape=[1] + self.output_dim_wo_batch
            )
            layer_backward = Cropping2D(
                cropping=(0, padding), data_format=data_format
            )
            layer_reshape_a = Reshape(target_shape=self.input_dim_wo_batch)

            model_backward = Sequential(
                [layer_reshape_b, layer_backward, layer_reshape_a]
            )
            _ = model_backward(Input(self.output_dim_wo_batch))
            self.layer_backward = model_backward


def get_backward_ZeroPadding1D(layer: ZeroPadding1D) -> Layer:
    """
    This function creates a `BackwardZeroPadding1D` layer based on a given `ZeroPadding1D` layer. It provides
    a convenient way to obtain a backward approximation of the input `ZeroPadding1D` layer, using the
    `BackwardZeroPadding1D` class to reverse the zero padding operation.

    ### Parameters:
    - `layer`: A Keras `ZeroPadding1D` layer instance. The function uses this layer's configurations to set up the `BackwardZeroPadding1D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardZeroPadding1D`, which acts as the reverse layer for the given `ZeroPadding1D`.

    ### Example Usage:
    ```python
    from keras.layers import ZeroPadding1D
    from keras_custom.backward import get_backward_ZeroPadding1D

    # Assume `zero_padding_layer` is a pre-defined ZeroPadding1D layer
    backward_layer = get_backward_ZeroPadding1D(zero_padding_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardZeroPadding1D(layer)
