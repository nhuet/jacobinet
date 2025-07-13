import keras.ops as K
import numpy as np

FGSM = "fgsm"
PGD = "pgd"


def clip_lp_ball(x, x_center, radius, p):
    if p == np.inf:
        # project on l_inf norm: equivalent to clipping
        return K.clip(x, x_center - radius, x_center + radius)

    axis = tuple(np.arange(len(x.shape) - 1) + 1)
    if p == 2:
        # compute l2 norm and normalize with it
        norm_2 = K.sum(K.sqrt((x - x_center) ** 2), axis=axis, keepdims=True)
        return K.clip(
            x, -radius * K.abs(x - x_center) / norm_2, radius * K.abs(x_center - x) / norm_2
        )
    if p == 1:
        # compute l1 norm and normalize with it
        norm_1 = K.sum(K.abs(x - x_center), axis=axis, keepdims=True)
        return K.clip(
            x, -radius * K.abs(x_center - x) / norm_1, radius * K.abs(x_center - x) / norm_1
        )
    else:
        raise ValueError("unknown lp norm p={}".format(p))
