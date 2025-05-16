from typing import List, Optional

import keras
import keras.ops as K  # type: ignore
from jacobinet.layers.merging.base_merge import BackwardMergeNonLinearLayer
from keras import KerasTensor as Tensor  # type: ignore
from keras.layers import Layer, Minimum  # type: ignore


@keras.saving.register_keras_serializable()
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
        self,
        gradient: Tensor,
        input: List[Tensor],
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        output_layer = self.layer(input)
        return [gradient * (K.sign(output_layer - input_i) + 1) for input_i in input]


def get_backward_Minimum(layer: Minimum) -> Layer:
    """
    This function creates a `BackwardMinimum` layer based on a given `Minimum` layer. It provides
    a convenient way to obtain the backward pass of the input `Minimum` layer, using the
    `BackwardMinimum`.

    ### Parameters:
    - `layer`: A Keras `Minimum` layer instance. The function uses this layer's configurations to set up the `BackwardMinimum` layer.

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
