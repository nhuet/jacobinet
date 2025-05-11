from typing import Iterable, List, Tuple, Union

from keras import KerasTensor as Tensor


def to_list(tensor: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """
    Converts a single tensor or a list of tensors to a list of tensors.

    If the input is already a list of tensors, it returns the list unchanged.
    If the input is a single tensor, it wraps it in a list.

    Args:
        tensor: A single tensor or a list of tensors.

    Returns:
        List[Tensor]: A list containing the input tensor(s).

    Example:
        # Single tensor
        tensor = tf.constant([1, 2, 3])
        tensor_list = to_list(tensor)
        print(tensor_list)  # Output: [tensor]

        # List of tensors
        tensor_list = to_list([tensor, tensor])
        print(tensor_list)  # Output: [tensor, tensor]
    """
    if isinstance(tensor, list):
        return tensor
    return [tensor]


def to_tuple(x: Union[int, Iterable[int]]) -> Tuple[int, ...]:
    """
    Convert an integer or iterable of integers into a tuple of integers.

    This utility function ensures that the input is returned as a tuple of integers,
    regardless of whether a single integer or an iterable (like a list or tuple) of integers is provided.

    Parameters
    ----------
    x : int or Iterable[int]
        The input value to be converted. If an `int` is provided, it is wrapped in a tuple.
        If an iterable of integers is provided, it is converted to a tuple.

    Returns
    -------
    Tuple[int, ...]
        A tuple of integers.

    Examples
    --------
    >>> to_tuple(5)
    (5,)

    >>> to_tuple([1, 2, 3])
    (1, 2, 3)
    """
    return (x,) if isinstance(x, int) else tuple(x)
