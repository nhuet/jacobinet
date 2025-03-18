from typing import Callable, Union

import keras  # type: ignore
import keras.ops as K  # type: ignore
from keras import KerasTensor as Tensor  # type: ignore

RELU = "relu"
RELU6 = "relu6"
LEAKY_RELU = "leaky_relu"
ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
SIGMOID = "sigmoid"
TANH = "tanh"
SILU = "silu"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
HARD_SILU = "hard_silu"
LINEAR = "linear"
MISH = "mish"
SWISH = "swish"
HARD_SWISH = "hard_swish"


def deserialize(name: str) -> Callable[..., list[Tensor]]:
    """Get the activation from name.

    Args:
        name: name of the method.
    among the implemented Keras activation function.

    Returns:
        the activation function
    """
    name = name.lower()
    if name == RELU:
        return relu_prime
    elif name == RELU6:
        return relu6_prime
    elif name == LEAKY_RELU:
        return leaky_relu_prime
    elif name == ELU:
        return elu_prime
    elif name == SELU:
        return selu_prime
    elif name == SOFTPLUS:
        return softplus_prime
    elif name == SOFTSIGN:
        return softsign_prime
    elif name == SOFTMAX:
        return softmax_prime
    elif name == SIGMOID:
        return sigmoid_prime
    elif name == TANH:
        return tanh_prime
    elif name == SILU:
        return silu_prime
    elif name == EXPONENTIAL:
        return exponential_prime
    elif name == HARD_SIGMOID:
        return hard_sigmoid_prime
    elif name == HARD_SILU:
        return hard_silu_prime
    elif name == LINEAR:
        return linear_prime
    elif name == MISH:
        return mish_prime
    elif name == SWISH:
        return silu_prime
    elif name == HARD_SWISH:
        return hard_silu_prime


def softmax_prime(inputs: Tensor):
    """
    Computes the derivative of the Softmax activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Mish derivative for.

    Returns:
    - Derivative of Softmax with respect to the inputs.
    """
    raise UserWarning(
        "Softmax is unstable for verification and should be avoided. Consider using logits to improve robustness against misclassification."
    )
    raise NotImplementedError()


# do a custom gradient because of non continuity
# @keras.ops.custom_gradient
def relu_prime(
    inputs: Tensor,
    negative_slope: float = 0.0,
    threshold: float = 0.0,
    max_value: Union[None, float] = None,
) -> Tensor:
    """
    Computes the derivative of the Rectified Linear Unit (ReLU) activation function.

    The ReLU derivative has a slope of 1 for positive inputs above the threshold and a slope of `negative_slope` for inputs below the threshold.
    When `max_value` is provided, the derivative is zero for values exceeding `max_value`.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the ReLU derivative for.
    - negative_slope: The slope for negative input values (default is 0, making ReLU zero for negative inputs).
    - threshold: Threshold value where the ReLU becomes active (default is 0).
    - max_value: Upper limit beyond which the derivative is zero if `max_value` is specified (default is None).

    Returns:
        Derivative of ReLU with respect to the inputs.
    """

    mask_neg_slope: Tensor = K.relu(K.sign(threshold - inputs))  # 1 iff layer_input[i]<threhshold
    mask_value: Tensor = K.relu(K.sign(inputs - threshold))
    if max_value:
        mask_value *= K.relu(K.sign(max_value - inputs))

    backward_relu: Tensor = negative_slope * mask_neg_slope + (1 - mask_neg_slope) * mask_value

    def grad(*args, upstream=None):
        return keras.ops.sigmoid(upstream)

    return backward_relu  # , grad


def relu6_prime(inputs: Tensor, negative_slope: float = 0.0, threshold: float = 0.0) -> Tensor:
    """
    Computes the derivative of the ReLU6 activation function, which is a clipped version of ReLU.

    This function uses `relu_prime` with a `max_value` of 6, providing a capped ReLU activation.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the ReLU6 derivative for.
    - negative_slope: The slope for negative input values (default is 0).
    - threshold: Threshold value where the ReLU6 becomes active (default is 0).

    Returns:
        Derivative of ReLU6 with respect to the inputs.
    """
    return relu_prime(
        inputs,
        negative_slope=negative_slope,
        threshold=threshold,
        max_value=6.0,
    )


def leaky_relu_prime(inputs: Tensor, negative_slope: float = 0.2) -> Tensor:
    """
    Computes the derivative of the Leaky ReLU activation function.

    Leaky ReLU has a small positive slope for negative inputs, allowing a non-zero gradient when inputs are below zero.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Leaky ReLU derivative for.
    - negative_slope: The slope for negative input values (default is 0.2).

    Returns:
    - Derivative of Leaky ReLU with respect to the inputs.
    """
    return relu_prime(inputs, negative_slope=negative_slope)


def elu_prime(inputs: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Computes the derivative of the Exponential Linear Unit (ELU) activation function.

    The ELU activation is linear for positive inputs and exponential for negative inputs with a scaling factor `alpha`.
    The derivative of elu with `alpha > 0` is defined as:

    - `x > 0` => elu'(x)=1
    - `x < 0` => elu'(x) = alpha*exp(x)

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the ELU derivative for.
    - alpha: Scaling factor for negative inputs (default is 1).

    Returns:
    - Derivative of ELU with respect to the inputs.
    """
    elu_prime_pos = K.relu(K.sign(inputs))
    elu_prime_neg = (K.elu(inputs, alpha) + 1) * (1 - elu_prime_pos)

    return elu_prime_pos + elu_prime_neg


def selu_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Scaled Exponential Linear Unit (SELU) activation function.

    SELU scales the ELU activation with fixed parameters `alpha` and `scale` to induce self-normalizing properties.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the SELU derivative for.

    Returns:
    - Derivative of SELU with respect to the inputs.
    """
    alpha: float = 1.67326324
    scale: float = 1.05070098

    return scale * elu_prime(inputs, alpha=alpha)


def softplus_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Softplus activation function.

    Softplus has a smooth gradient, often used as a differentiable alternative to ReLU.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Softplus derivative

    Returns:
        Derivative of Softplus with respect to the inputs.
    """
    return K.exp(inputs) / (K.exp(inputs) + 1)


def softsign_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Softsign activation function.

    Softsign provides a smooth gradient with a bounded output range between -1 and 1.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Softsign derivative for.

    Returns:
        Derivative of Softsign with respect to the inputs.
    """
    return 1.0 / (K.abs(inputs) + 1) ** 2


def sigmoid_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Sigmoid activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Sigmoid derivative for.

    Returns:
    - Derivative of Sigmoid with respect to the inputs.
    """
    sigmoid_x: Tensor = K.sigmoid(inputs)
    return sigmoid_x * (1 - sigmoid_x)


def tanh_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Tanh activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Tanh derivative for.

    Returns:
    - Derivative of Tanh with respect to the inputs.
    """
    return 1 - K.tanh(inputs) ** 2


def silu_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the SiLU activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the SiLU derivative for.

    Returns:
    - Derivative of SiLU with respect to the inputs.
    """
    return K.sigmoid(inputs) + inputs * sigmoid_prime(inputs)


def exponential_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Exponential activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Exponential derivative for.

    Returns:
    - Derivative of Exponential with respect to the inputs.
    """
    return K.exp(inputs)


def hard_sigmoid_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Hard Sigmoid activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Hard Sigmoid derivative for.

    Returns:
    - Derivative of Hard Sigmoid with respect to the inputs.
    """

    mask_strictly_lower_than_3: Tensor = K.relu(K.sign(3 - inputs))  # 1 if inputs <3
    mask_strictly_greater_than_minus_3: Tensor = K.relu(K.sign(inputs + 3))  # 1 if inputs >-3

    return mask_strictly_greater_than_minus_3 * mask_strictly_lower_than_3 / 6.0


def hard_silu_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Hard SiLU activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Hard SiLU derivative for.

    Returns:
    - Derivative of Hard SiLU with respect to the inputs.
    """

    mask_greater_than_minus_3: Tensor = K.relu(
        K.sign(inputs + 3 + keras.backend.epsilon())
    )  # 1 if inputs >= -3
    mask_lower_than_3: Tensor = K.relu(
        K.sign(3 + keras.backend.epsilon() - inputs)
    )  # 1 if inputs <=3

    return (1 - mask_lower_than_3) + (
        inputs / 3.0 + 0.5
    ) * mask_lower_than_3 * mask_greater_than_minus_3


def linear_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Linear activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Linear derivative for.

    Returns:
    - Derivative of Linear with respect to the inputs.
    """
    return K.ones_like(inputs)


def mish_prime(inputs: Tensor) -> Tensor:
    """
    Computes the derivative of the Mish activation function.

    Please refer to the official Keras documentation for additional information.

    Parameters:
    - inputs: The input tensor to compute the Mish derivative for.

    Returns:
    - Derivative of Mish with respect to the inputs.
    """

    # mish = x*tanh(softplus(x))
    # mish'(x) = tanh(softplus(x)) + x*softplus'(x)*tanh'(softplus(x))
    softplus_x = K.softplus(inputs)

    return K.tanh(softplus_x) + inputs * softplus_prime(inputs) * tanh_prime(softplus_x)
