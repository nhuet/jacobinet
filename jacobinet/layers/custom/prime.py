import keras.ops as K

def max_prime(inputs, axis):

    indices = K.argmax(inputs, axis)
    return K.one_hot(indices, 4, axis=axis)

