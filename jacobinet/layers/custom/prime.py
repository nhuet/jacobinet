import keras.ops as K


def max_prime(inputs, axis):

    axis_dim = inputs.shape[axis]
    output = K.max(inputs, axis=axis, keepdims=True)
    output_sampling = K.repeat(output, axis_dim, axis)
    # 1 if contribute to the max
    return K.relu(
        2 * K.sign(inputs - output_sampling) + 1
    )  # 1 at coordinate i iff inputs[i] == output_sampling[i]
