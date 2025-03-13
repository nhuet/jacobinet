from typing import Union, List
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
