from typing import Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.layer import BackwardLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer  # type: ignore
from keras.layers import UpSampling3D  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardUpSampling3D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `UpSampling3D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the upsampling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import UpSampling3D
    from keras_custom.backward.layers import BackwardUpSampling3D

    # Assume `upsampling_layer` is a pre-defined UpSampling3D layer
    backward_layer = BackwardUpSampling3D(upsampling_layer)
    output = backward_layer(input_tensor)
    """

    layer: UpSampling3D

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # If data_format is "channels_last": (batch_size, rows, cols, channels)
        if self.layer.data_format == "channels_last":
            # (batch_size, W, H, D, C)
            W, H, D, C = self.input_dim_wo_batch
            gradient = K.sum(
                K.reshape(
                    gradient,
                    [
                        -1,
                        W,
                        self.layer.size[0],
                        H,
                        self.layer.size[1],
                        D,
                        self.layer.size[2],
                        C,
                    ],
                ),
                (2, 4, 6),
            )
        else:
            C, W, H, D = self.input_dim_wo_batch
            gradient = K.sum(
                K.reshape(
                    gradient,
                    [
                        -1,
                        C,
                        W,
                        self.layer.size[0],
                        H,
                        self.layer.size[1],
                        D,
                        self.layer.size[2],
                    ],
                ),
                (3, 5, 7),
            )

        return gradient


def get_backward_UpSampling3D(layer: UpSampling3D) -> Layer:
    """
    This function creates a `BackwardUpSampling3D` layer based on a given `UpSampling3D` layer. It provides
    a convenient way to obtain a backward approximation of the input `UpSampling3D` layer, using the
    `BackwardUpSampling3D` class to reverse the upsampling operation.

    ### Parameters:
    - `layer`: A Keras `UpSampling3D` layer instance. The function uses this layer's configurations to set up the `BackwardUpSampling3D` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardUpSampling3D`, which acts as the reverse layer for the given `UpSampling3D`.

    ### Example Usage:
    ```python
    from keras.layers import UpSampling3D
    from keras_custom.backward import get_backward_UpSampling3D

    # Assume `upsampling_layer` is a pre-defined UpSampling3D layer
    backward_layer = get_backward_UpSampling3D(zero_padding_layer)
    output = backward_layer(input_tensor)
    """

    return BackwardUpSampling3D(layer)
