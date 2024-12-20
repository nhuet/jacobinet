from keras.layers import Input, Layer
from keras.losses import Loss

from jacobinet.attacks import AdvModel
from jacobinet.attacks.fgsm import get_fgsm_model
from jacobinet.attacks.pgd import get_pgd_model
from .utils import FGSM, PGD

from typing import Union, Callable


default_mapping_attack: dict[str, type[Callable]] = { # type: ignore
    FGSM: get_fgsm_model,
    PGD: get_pgd_model,
}


def get_adv_model(model, 
                  loss:Union[str, Loss, Layer]='categorical_crossentropy', 
                  attack:str=FGSM,
                  mapping_keras2backward_classes={},
                  mapping_keras2backward_losses={},
                  mapping_attack={},# in case the user wants to define its own attack
                  **kwargs)->AdvModel:
    
    if mapping_attack is not None:
        default_mapping_attack.update(mapping_attack)

    get_method_adv = default_mapping_attack.get(attack)
    if get_method_adv is None:
        raise ValueError(
            "The mapping from the an attack is not available in default_mapping_attack, {} not found".format(
                attack
            )
        )
    
    return get_method_adv(model, 
                  loss=loss,
                  mapping_keras2backward_classes=mapping_keras2backward_classes,
                  mapping_keras2backward_losses=mapping_keras2backward_losses,
                  **kwargs)
