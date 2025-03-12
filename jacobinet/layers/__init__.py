from .convolutional import (
    get_backward_Conv2D,
    get_backward_Conv1D,
    get_backward_DepthwiseConv2D,
    get_backward_DepthwiseConv1D,
)

# from .custom import get_backward_MulConstant, get_backward_PlusConstant
from .normalization import get_backward_BatchNormalization
from .reshaping import (
    get_backward_ZeroPadding2D,
    get_backward_ZeroPadding3D,
    get_backward_ZeroPadding1D,
    get_backward_Cropping3D,
    get_backward_Cropping1D,
    get_backward_Cropping2D,
    get_backward_Reshape,
    get_backward_Flatten,
    get_backward_Permute,
    get_backward_RepeatVector,
    get_backward_UpSampling2D,
    get_backward_UpSampling1D,
    get_backward_UpSampling3D
)
from .pooling import (
    get_backward_AveragePooling2D,
    get_backward_AveragePooling3D,
    get_backward_AveragePooling1D,
    get_backward_GlobalAveragePooling2D,
    get_backward_GlobalAveragePooling1D,
    get_backward_GlobalAveragePooling3D,
    get_backward_MaxPooling2D,
    get_backward_GlobalMaxPooling2D,
)

from .core import (
    get_backward_Dense,
    get_backward_EinsumDense,
    get_backward_Activation,
)


from .activations import (
    get_backward_ReLU,
    get_backward_LeakyReLU,
    get_backward_PReLU,
    get_backward_ELU,
)

from .merging import (
    BackwardMergeNonLinearLayer,
    BackwardAdd,
    get_backward_Add,
    get_backward_Average,
    get_backward_Concatenate,
    get_backward_Maximum,
    get_backward_Minimum,
    get_backward_Subtract,
    get_backward_Multiply,
)

from .layer import BackwardLayer, BackwardLinearLayer, BackwardNonLinearLayer, BackwardBoundedLinearizedLayer

from jacobinet.layers.convolutional import (
    BackwardConv3D,
    BackwardConv2D,
    BackwardConv1D,
    BackwardDepthwiseConv2D,
    BackwardDepthwiseConv1D,
)

from jacobinet.layers.core import (
    BackwardDense,
    BackwardEinsumDense
)

from jacobinet.layers.normalization import BackwardBatchNormalization

from jacobinet.layers.pooling import (
    BackwardAveragePooling1D,
    BackwardAveragePooling2D,
    BackwardAveragePooling3D,
    BackwardGlobalAveragePooling1D,
    BackwardGlobalAveragePooling2D,
    BackwardGlobalAveragePooling3D
)

from jacobinet.layers.reshaping import (
    BackwardZeroPadding1D,
    BackwardZeroPadding2D,
    BackwardZeroPadding3D,
    BackwardCropping1D,
    BackwardCropping2D,
    BackwardCropping3D,
    BackwardFlatten,
    BackwardRepeatVector,
    BackwardReshape,
    BackwardPermute,
    BackwardUpSampling1D,
    BackwardUpSampling2D,
    BackwardUpSampling3D,
)
