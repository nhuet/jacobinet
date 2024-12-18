from typing import List, Union
from keras.layers import Layer, ZeroPadding2D, Cropping2D, ZeroPadding1D, Cropping1D, ZeroPadding3D, Cropping3D, Permute, Input
import keras.ops as K


# compute output shape post convolution
def compute_output_pad(input_shape_wo_batch, kernel_size, strides, padding, data_format):
    if data_format == "channels_first":
        w, h = input_shape_wo_batch[1:]
    else:
        w, h = input_shape_wo_batch[:-1]
    k_w, k_h = kernel_size
    if padding == "same":
        p = 0
    s_w, s_h = strides

    w_pad = (w - k_w + 2 * p) / s_w + 1 - w
    h_pad = (h - k_h + 2 * p) / s_h + 1 - h
    return (w_pad, h_pad)


def pooling_layer2D(w_pad, h_pad, data_format) -> List[Union[ZeroPadding2D, Cropping2D]]:
    if w_pad or h_pad:
        # add padding
        if w_pad >= 0 and h_pad >= 0:
            padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (h_pad // 2, h_pad // 2 + h_pad % 2))
            pad_layer = [ZeroPadding2D(padding, data_format=data_format)]
        elif w_pad <= 0 and h_pad <= 0:
            w_pad *= -1
            h_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (h_pad // 2, h_pad // 2 + h_pad % 2))
            pad_layer = [Cropping2D(cropping, data_format=data_format)]
        elif w_pad > 0 and h_pad < 0:
            h_pad *= -1
            padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            cropping = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        else:
            w_pad *= -1
            padding = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        return pad_layer
    return []


def pooling_layer1D(w_pad, data_format) -> List[Union[ZeroPadding1D, Cropping1D]]:
    if w_pad:
        # add padding
        if w_pad >= 0:
            padding = (w_pad // 2, w_pad // 2 + w_pad % 2)
            pad_layer = [ZeroPadding1D(padding, data_format=data_format)]
        else:
            w_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (w_pad // 2, w_pad // 2 + w_pad % 2)
            if data_format == "channels_first":
                # we need to permute the layer to apply Cropping on the right dimensions
                perm_layer = Permute((2, 1))
                pad_layer = [perm_layer, Cropping1D(cropping), perm_layer]
            else:
                pad_layer = [Cropping1D(cropping)]
        return pad_layer
    return []


def pooling_layer3D(d_pad, w_pad, h_pad, data_format) -> List[Union[ZeroPadding3D, Cropping3D]]:
    if d_pad or w_pad or h_pad:
        # add padding
        if d_pad >= 0 and w_pad >= 0 and h_pad >= 0:
            padding = (
                (d_pad // 2, d_pad // 2 + d_pad % 2),
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [ZeroPadding3D(padding, data_format=data_format)]
        elif d_pad <= 0 and w_pad <= 0 and h_pad <= 0:
            d_pad *= -1
            w_pad *= -1
            h_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (
                (d_pad // 2, d_pad // 2 + d_pad % 2),
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [Cropping3D(cropping, data_format=data_format)]
        elif w_pad > 0 and h_pad < 0:
            h_pad *= -1
            padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            cropping = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        else:
            w_pad *= -1
            padding = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
            cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
            pad_layer = [
                ZeroPadding2D(padding, data_format=data_format),
                Cropping2D(cropping, data_format=data_format),
            ]
        return pad_layer
    else:
        return []


def call_backward_depthwise2d(inputs, layer, op_reshape, op_split, inner_models, axis, axis_c, c_in, use_bias):
    # remove bias if needed
    if hasattr(layer, "use_bias") and layer.use_bias and use_bias:
        if layer.data_format == "channels_first":
            inputs = inputs - layer.bias[None, :, None, None]  # (batch, d_m*c_in, w_out, h_out)
        else:
            inputs = inputs - layer.bias[None, None, None, :]  # (batch, w_out, h_out, d_m*c_in)

    outputs = op_reshape(inputs)  # (batch, d_m, c_in, w_out, h_out) if data_format=channel_first

    # if self.layer.use_bias and self.use_bias:

    split_outputs = K.split(outputs, c_in, axis=axis_c)  # [(batch, d_m, 1, w_out, h_out)]
    split_outputs = [op_split(s_o_i) for s_o_i in split_outputs]  # [(batch_size, d_m, w_out, h_out)]

    conv_outputs = [inner_models[i](s_o_i) for (i, s_o_i) in enumerate(split_outputs)]  # [(batch_size, 1, w_in, h_in)]
    return K.concatenate(conv_outputs, axis=axis)  # (batch_size, c_in, w_in, h_in)


def reshape_to_batch(input_tensor, layer_output_shape):
    if len(input_tensor.shape)!= len(layer_output_shape):
        # backward is not in a diagonal format, current shape is [batch_size]+n_out+layer_output_shape
        # we reshape it to [batch_size*np.prod(n_out_)]+layer_output_shape
        n_out = list(input_tensor.shape[:-len(layer_output_shape[1:])][1:])
        return True, K.reshape(input_tensor, [-1]+layer_output_shape[1:]), n_out
    else:
        return False, input_tensor, []
    
# source: decomon/keras_utils.py
# author: nolwen huet
def share_weights_and_build(original_layer: Layer, new_layer: Layer, weight_names: list[str], input_shape_wo_batch:list[int]) -> None:
    """Share the weights specidifed by names of an already built layer to another unbuilt layer.

    We assume that each weight is also an original_laer's attribute whose name is the weight name.

    Args:
        original_layer: the layer used to share the weights
        new_layer: the new layer which will be buit and will share the weights of the original layer
        weight_names: names of the weights to share

    Returns:

    """
    # Check the original_layer is built and the new_layer is not built
    if not original_layer.built:
        raise ValueError("The original layer must already be built for sharing its weights.")
    if new_layer.built:
        raise ValueError("The new layer must not be built to get the weights of the original layer")

    # store the weights as a new_layer variable before build (ie before the lock)
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        try:
            setattr(new_layer, w_name, w)
        except AttributeError:
            # manage hidden weights introduced for LoRA https://github.com/keras-team/keras/pull/18942
            w_name = f"_{w_name}"
            w = getattr(original_layer, w_name)
            setattr(new_layer, w_name, w)

    # build the layer
    new_layer(Input(input_shape_wo_batch))
    # overwrite the newly generated weights and untrack them
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        w_to_drop = getattr(new_layer, w_name)
        try:
            setattr(new_layer, w_name, w)
        except AttributeError:
            # manage hidden weights introduced for LoRA https://github.com/keras-team/keras/pull/18942
            w_name = f"_{w_name}"
            w = getattr(original_layer, w_name)
            w_to_drop = getattr(new_layer, w_name)
            setattr(new_layer, w_name, w)
        # untrack the not used anymore weight
        new_layer._tracker.untrack(w_to_drop)

