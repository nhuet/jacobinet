import math
from typing import Any, List, Optional, Union

from keras.layers import ZeroPadding2D  # type: ignore
from keras.layers import Input, Reshape
from keras.models import Model
from keras.src import ops  # type: ignore

from .utils_conv import get_conv_op


class ConstantPadding2D(ZeroPadding2D):
    """constant-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of constant value at the top, bottom, left and
    right side of an image tensor.

    Example:

    >>> input_shape = (1, 1, 2, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[[0 1]
       [2 3]]]]
    >>> y = keras.layers.ConstantPadding2D(const=5, padding=1)(x)
    >>> y
    [[[[5 5]
       [5 5]
       [5 5]
       [5 5]]
      [[5 5]
       [0 1]
       [2 3]
       [5 5]]
      [[5 5]
       [5 5]
       [5 5]
       [5 5]]]]

    Args:
        padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding is applied to height and width.
            - If tuple of 2 ints: interpreted as two different symmetric padding
              values for height and width:
              `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints: interpreted as
             `((top_pad, bottom_pad), (left_pad, right_pad))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, height, width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, padded_height, padded_width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, padded_height, padded_width)`
    """

    def __init__(
        self,
        const: Union[float, int] = 0.0,
        padding: tuple[int, ...] = (1, 1),
        data_format: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(padding=padding, data_format=data_format, **kwargs)
        self.const = const

    def call(self, inputs):
        if self.data_format == "channels_first":
            all_dims_padding = ((0, 0), (0, 0), *self.padding)
        else:
            all_dims_padding = ((0, 0), *self.padding, (0, 0))
        return ops.pad(
            inputs, all_dims_padding, constant_values=self.const
        )  # , mode="constant", constant_values=self.const)

    def get_config(self):
        config = {"const": self.const}
        base_config = super().get_config()
        return {**base_config, **config}


def get_pad_width(
    input_shape_wo_batch: List[int],
    pool_size: List[int],
    strides: List[int],
    data_format: str,
) -> List[List[int]]:
    if data_format == "channels_first":
        input_shape_wh = input_shape_wo_batch[1:]
    else:
        input_shape_wh = input_shape_wo_batch[:2]

    # width
    # shape with valid
    output_shape_w_valid = math.floor((input_shape_wh[0] - pool_size[0]) / strides[0]) + 1
    # shape with same
    output_shape_w_same = math.floor((input_shape_wh[0] - 1) / strides[0]) + 1

    output_shape_h_valid = math.floor((input_shape_wh[1] - pool_size[1]) / strides[1]) + 1
    # shape with same
    output_shape_h_same = math.floor((input_shape_wh[1] - 1) / strides[1]) + 1

    w_pad: int = output_shape_w_same - output_shape_w_valid
    h_pad: int = output_shape_h_same - output_shape_h_valid
    # split in top, bottom, left, right
    pad_top: int = w_pad // 2
    pad_bottom: int = pad_top + w_pad % 2
    pad_left: int = h_pad // 2
    pad_right: int = pad_left + h_pad % 2

    pad_batch = [0, 0]
    pad_channel = [0, 0]
    pad_width = [pad_top, pad_bottom]
    pad_height = [pad_left, pad_right]

    padding = [pad_width, pad_height]

    if data_format == "channels_first":
        # pad_width = [pad_batch, pad_channel, pad_width, pad_height]
        pad_width = [pad_batch, pad_channel] + padding
    else:
        # pad_width = [pad_batch, pad_width, pad_height, pad_channel]
        pad_width = [pad_batch] + padding + [pad_channel]

    return pad_width, padding


def get_linear_block_max(layer, input_dim_wo_batch):
    conv_op, _ = get_conv_op(layer, input_dim_wo_batch)
    # if padding = same, we need to add the padding manually as it is not the same padding in convolution
    # in that case padding is constant with padding_valud -np.inf
    if layer.padding == "same":
        # compute pad_width
        pad_width, padding = get_pad_width(
            input_shape_wo_batch=input_dim_wo_batch,
            pool_size=layer.pool_size,
            strides=layer.strides,
            data_format=layer.data_format,
        )

    x = Input(input_dim_wo_batch)

    if layer.padding == "same":
        padding_layer = ConstantPadding2D(
            const=-1e10,
            padding=padding,
            data_format=layer.data_format,
        )
        x_0 = padding_layer(x)
    else:
        x_0 = x
    x_1 = conv_op(x_0)
    conv_out_shape_wo_batch = list(x_1.shape[1:])

    if layer.data_format == "channels_first":
        in_channel = input_dim_wo_batch[0]
        image_shape = conv_out_shape_wo_batch[-2:]
        # image_shape = self.output_dim_wo_batch[-2:]
        target_shape = [in_channel, -1] + image_shape
    else:
        in_channel = input_dim_wo_batch[-1]
        image_shape = conv_out_shape_wo_batch[:2]
        # image_shape = self.output_dim_wo_batch[:2]
        target_shape = image_shape + [in_channel, -1]

    reshape_op = Reshape(target_shape)
    # compile a model to init shape
    # inner_model = Sequential([self.conv_op, self.reshape_op])
    x_2 = reshape_op(x_1)

    return Model(x, x_2)
