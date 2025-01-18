import keras.ops as K
from keras import KerasTensor as Tensor
from typing import List

def max_prime(inputs, axis:int):

    indices = K.argmax(inputs, axis)
    dim_i = inputs.shape[axis]
    return K.one_hot(indices, dim_i, axis=axis)

def global_max_prime(inputs, axis:List[int]):

    gradient = K.ones_like(inputs)
    for axis_ in axis:
        gradient_i = max_prime(inputs, axis_)
        gradient= gradient*gradient_i

    return gradient


