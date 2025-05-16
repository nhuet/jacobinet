from typing import List, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.merging.base_merge import BackwardMergeNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer, Maximum  # type: ignore


@keras.saving.register_keras_serializable()
class BackwardMaximum(BackwardMergeNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Maximum` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Maximum
    from keras_custom.backward.layers import BackwardMaximum

    # Assume `maximum_layer` is a pre-defined Maximum layer
    backward_layer = BackwardMaximum(maximum_layer)
    output = backward_layer(input_tensor)
    """

    def call_on_reshaped_gradient(
        self,
        gradient: Tensor,
        input: List[Tensor],
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        output_layer = self.layer(input)
        return [gradient * (K.sign(input_i - output_layer) + 1) for input_i in input]


def get_backward_Maximum(layer: Maximum) -> Layer:
    """
    This function creates a `BackwardMaximum` layer based on a given `Maximum` layer. It provides
    a convenient way to obtain the backward pass of the input `Maximum` layer, using the
    `BackwardMaximum`.

    ### Parameters:
    - `layer`: A Keras `Maximum` layer instance. The function uses this layer's configurations to set up the `BackwardMaximum` layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMaximum`, which acts as the reverse layer for the given `Maximum`.

    ### Example Usage:
    ```python
    from keras.layers import Maximum
    from keras_custom.backward import get_backward_Maximum

    # Assume `maximum_layer` is a pre-defined Maximum layer
    backward_layer = get_backward_Maximum(concat_layer)
    output = backward_layer(input_tensor)
    """
    return BackwardMaximum(layer)
