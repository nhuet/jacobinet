import keras.ops as K  # type: ignore
from keras import KerasTensor as Tensor
from typing import List


def max_prime(inputs: Tensor, axis: int):
    """
    Computes the derivative of the max operation with respect to the inputs along a specified axis.

    This function identifies the indices of the maximum values along the given axis and
    creates a one-hot encoded tensor at those indices. The result represents the derivative
    of the max function, where a '1' is placed at the position of the maximum value
    and '0's elsewhere.

    Args:
        inputs (tensor): The input tensor for which the derivative of the max operation is computed.
        axis (int): The axis along which to compute the maximum.

    Returns:
        tensor: A tensor of the same shape as `inputs`, with '1's at the positions of the maximum values
                along the specified axis and '0's elsewhere.

    Example:
        inputs = [[1, 2, 3], [4, 5, 6]]
        axis = 1
        max_prime(inputs, axis) = [[0, 0, 1], [0, 0, 1]]  # Returns a one-hot encoded tensor indicating the maximum indices.
    """

    indices = K.argmax(inputs, axis)
    dim_i = inputs.shape[axis]
    return K.one_hot(indices, dim_i, axis=axis)


def global_max_prime(inputs: Tensor, axis: List[int]):
    """
    Computes the derivative of the global max operation with respect to the inputs along multiple axes.

    This function calculates the derivative of the global maximum by applying the derivative of the
    max operation along each axis specified in `axis`. The result is a tensor where each element represents
    the derivative of the global max with respect to its corresponding input value.

    Args:
        inputs (tensor): The input tensor for which the derivative of the global max operation is computed.
        axis (List[int]): A list of axes along which the max operation is computed. Each axis will be used
                          to compute the derivative in a cumulative manner.

    Returns:
        tensor: A tensor representing the cumulative derivative of the global max operation across all axes
                specified in `axis`.

    Example:
        inputs = [[1, 2, 3], [4, 5, 6]]
        axis = [0, 1]
        global_max_prime(inputs, axis)= [[0,0, 0], [0, 0, 1]]  # Returns the derivative of the global max across the given axes.
    """

    gradient = K.ones_like(inputs)
    for axis_ in axis:
        gradient_i = max_prime(inputs, axis_)
        gradient = gradient * gradient_i

    return gradient
