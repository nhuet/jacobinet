from typing import Dict, List, Union

import keras
import numpy as np
from jacobinet.attacks.base_attacks import AdvLayer, AdvModel, get_adv_model_base
from jacobinet.layers.layer import BackwardLayer
from jacobinet.losses import BackwardLoss
from jacobinet.models import BackwardModel
from jacobinet.utils import to_list
from keras import KerasTensor as Tensor  # type:ignore
from keras.layers import Layer  # type:ignore
from keras.losses import Loss  # type:ignore

from .utils import FGSM


class FastGradientSign(AdvLayer):
    """
    Fast Gradient Sign Method (FGSM) for generating adversarial examples.

    This class implements the Fast Gradient Sign Method (FGSM) for creating adversarial examples by adding perturbations
    to the input based on the gradients of the loss function with respect to the input. It creates perturbations with
    a specified magnitude (epsilon) and applies the attack based on the gradient sign of the loss function.

    The FGSM attack is commonly used to evaluate model robustness by generating adversarial samples and applying them
    during training or testing.

    Attributes:
        epsilon: The magnitude of the perturbation to be applied. Controls the strength of the adversarial attack.
        lower: The lower bound for the perturbation, ensuring the perturbed input remains within valid input ranges.
        upper: The upper bound for the perturbation, ensuring the perturbed input remains within valid input ranges.

    Example:
        adv_layer = FastGradientSign(epsilon=0.1)
        adv_example = adv_layer([input_image, gradient])

    """

    def __init__(
        self,
        epsilon=0.0,
        lower=-np.inf,
        upper=np.inf,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = keras.Variable(epsilon, trainable=False)

    # @keras.ops.custom_gradient
    def call(self, inputs, training=None, mask=None):
        # inputs = [x and \delta f(x)]
        x, grad_x = inputs[:2]

        """
        def grad(*args, upstream=None):
            import pdb

            pdb.set_trace()
            return keras.ops.tanh(upstream)
        """

        # project given lp norm
        adv_x = x + self.epsilon * keras.ops.sign(-grad_x)
        adv_x = self.project_lp_ball(adv_x)
        # import pdb; pdb.set_trace()
        if len(inputs) > 2:
            lower, upper = inputs[:2]
            return keras.ops.minimum(keras.ops.maximum(adv_x, lower), upper)
        else:
            return keras.ops.clip(adv_x, self.lower, self.upper)

    # saving
    def get_config(self):
        config = super().get_config()
        eps_config = keras.saving.serialize_keras_object(self.epsilon)
        config["epsilon"] = eps_config
        config["lower"] = self.lower
        config["upper"] = self.upper
        return config


def get_fgsm_model(
    model,
    loss: Union[str, Loss, Layer] = "categorical_crossentropy",
    mapping_keras2backward_classes: Dict[type[Layer], type[BackwardLayer]] = {},  # define type
    mapping_keras2backward_losses: Dict[type[Layer], type[BackwardLoss]] = {},
    **kwargs,
) -> AdvModel:  # we do not compute gradient on extra_inputs, loss should return (None, 1)
    """
    Creates an adversarial model using the Fast Gradient Sign Method (FGSM) for generating adversarial examples.

    This function takes a Keras model and modifies it to generate adversarial examples using the FGSM attack.
    It computes the adversarial examples by applying perturbations based on the gradients of the loss function with respect to the model inputs.
    The model is returned with an additional adversarial layer (`AdvLayer`) that applies the attack to the model's predictions.

    Parameters:
        model: The base Keras model to be used for generating adversarial examples.
        loss: The loss function to be used during the adversarial training or testing process. Default is 'categorical_crossentropy'.
        mapping_keras2backward_classes: A dictionary mapping Keras layer types to their corresponding backward layer types. Default is an empty dictionary.
        mapping_keras2backward_losses: A dictionary mapping loss functions to their corresponding backward loss types. Default is an empty dictionary.
        **kwargs: Additional keyword arguments, including:
            - 'lower': The lower bound for the adversarial perturbation (default: -np.inf).
            - 'upper': The upper bound for the adversarial perturbation (default: np.inf).
            - 'p': The Lp norm to use for the adversarial attack (default: -1).
            - 'radius': The radius for the perturbation (default: np.inf).
            - 'epsilon': The magnitude of the perturbation to apply during the FGSM attack.

    Returns:
        AdvModel: The model with an adversarial layer (`AdvLayer`) that applies the FGSM adversarial attack to the model's outputs.

    Raises:
        ValueError: If any of the expected arguments are of the wrong type or missing required parameters.

    Example:
        fgsm_model = get_fgsm_model(
            model=my_keras_model,
            loss='categorical_crossentropy',
            epsilon=0.1,
            lower=-1.0,
            upper=1.0,
            p=2,
            radius=0.2
        )

    In this example, the `get_fgsm_model` function is used to create an adversarial model using the FGSM attack.
    The adversarial perturbations will be applied to the model's inputs, and the resulting perturbed outputs will be used
    for adversarial training or testing.
    """
    base_adv_model: BackwardModel = get_adv_model_base(
        model=model,
        loss=loss,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
        mapping_keras2backward_losses=mapping_keras2backward_losses,
        **kwargs,
    )

    inputs: List[Tensor] = to_list(base_adv_model.inputs)
    adv_pred: List[Tensor] = to_list(base_adv_model.outputs)
    lower = -np.inf
    upper = np.inf
    p = -1
    radius = np.inf
    bounds = []
    if "lower" in kwargs and "upper" in kwargs:
        lower = kwargs["lower"]
        upper = kwargs["upper"]
        bounds = [lower, upper]
    if "p" in kwargs:
        p = kwargs["p"]
    if "radius" in kwargs:
        radius = kwargs["radius"]

    # use lp norm as well
    if "epsilon" in kwargs:
        fgsm_layer = FastGradientSign(
            epsilon=kwargs["epsilon"],
            p=p,
            radius=radius,
        )
    else:
        fgsm_layer = FastGradientSign(
            epsilon=kwargs["epsilon"],
            p=p,
            radius=radius,
        )

    output = fgsm_layer(inputs[:-1] + adv_pred + bounds)

    extra_inputs = []
    if "extra_inputs" in kwargs:
        extra_inputs = kwargs["extra_inputs"]

    fgsm_model = AdvModel(
        inputs=inputs + extra_inputs,
        outputs=output,
        layer_adv=fgsm_layer,
        backward_model=base_adv_model,
        method=FGSM,
    )

    return fgsm_model
