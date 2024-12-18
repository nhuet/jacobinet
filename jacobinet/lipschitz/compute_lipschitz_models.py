# from backward
from jacobinet.lipschitz import Lipschitz, LipschitzModel
from jacobinet.models import BackwardModel, BackwardSequential

from typing import Union


# from model
def get_lipschitz_model(backward_model:Union[BackwardModel, BackwardSequential], p:float):
    if not (isinstance(backward_model, BackwardModel) or isinstance(backward_model, BackwardSequential)):
        raise ValueError('the model is not a backward model')

    inputs = backward_model.input
    outputs = backward_model.output
    
    if isinstance(outputs, list):
        lip_output = [Lipschitz(p)(output_i) for output_i in outputs]
    else:
        lip_output = Lipschitz(p)(outputs)
    
    return LipschitzModel(inputs=inputs, outputs=lip_output, lipschitz_norm=p, backward_model=backward_model)
