from typing import Union, List
from keras import KerasTensor as Tensor


def to_list(tensor: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    if isinstance(tensor, list):
        return tensor
    return [tensor]
