import keras
from keras.src.layers.merging.base_merge import Merge
from jacobinet.layers.layer import BackwardLinearLayer, BackwardNonLinearLayer
from typing import Union, List


class BackwardMergeLinearLayer(BackwardLinearLayer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardMergeLinearLayer(layer)
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardMergeLinearLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """

    layer: Merge


class BackwardMergeNonLinearLayer(BackwardNonLinearLayer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardMergeLinearLayer(layer)
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardMergeLinearLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """

    layer: Merge
