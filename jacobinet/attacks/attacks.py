from typing import Union, Callable

from keras.layers import Layer  # type:ignore
from keras.losses import Loss  # type:ignore

from jacobinet.attacks import AdvModel
from jacobinet.attacks.fgsm import get_fgsm_model
from jacobinet.attacks.pgd import get_pgd_model
from .utils import FGSM, PGD


default_mapping_attack: dict[str, type[Callable]] = {  # type: ignore
    FGSM: get_fgsm_model,
    PGD: get_pgd_model,
}


def get_adv_model(
    model,
    loss: Union[str, Loss, Layer] = "categorical_crossentropy",
    attack: str = FGSM,
    mapping_keras2backward_classes={},
    mapping_keras2backward_losses={},
    mapping_attack={},  # in case the user wants to define its own attack
    **kwargs,
) -> AdvModel:
    """
    Creates an adversarial model by applying a specified attack method to the given model.

    Args:
        model: The original model to apply adversarial attacks to.
        loss: The loss function to use for the adversarial model. Can be a string, Loss class, or Layer. Default is 'categorical_crossentropy'.
        attack: The attack method to use. Default is FGSM (Fast Gradient Sign Method).
        mapping_keras2backward_classes: A dictionary mapping Keras layers to backward layer classes.
        mapping_keras2backward_losses: A dictionary mapping Keras loss classes to corresponding backward loss classes.
        mapping_attack: A dictionary allowing the user to define custom attack methods. Default is empty.
        **kwargs: Additional arguments passed to the attack function.

    Returns:
        AdvModel: The adversarial model with the specified attack applied.

    Raises:
        ValueError: If the specified attack method is not available in the `mapping_attack` dictionary.
    """

    if mapping_attack is not None:
        default_mapping_attack.update(mapping_attack)

    get_method_adv = default_mapping_attack.get(attack)
    if get_method_adv is None:
        raise ValueError(
            "The mapping from the an attack is not available in default_mapping_attack, {} not found".format(
                attack
            )
        )

    return get_method_adv(
        model,
        loss=loss,
        mapping_keras2backward_classes=mapping_keras2backward_classes,
        mapping_keras2backward_losses=mapping_keras2backward_losses,
        **kwargs,
    )
