# from backward
from typing import Union

from jacobinet.lipschitz import Lipschitz, LipschitzModel
from jacobinet.models import BackwardModel, BackwardSequential


# from model
def get_lipschitz_model(
    backward_model: Union[BackwardModel, BackwardSequential], p: float
) -> LipschitzModel:
    """
    Creates a `LipschitzModel` that computes the Lipschitz constant for the given backward model at a given input.

    This function wraps the provided `backward_model` with a `LipschitzModel`, where the output of
    the model is processed using the specified Lipschitz norm (`p`). It supports both single-output
    models and models with multiple outputs.

    Args:
        backward_model: The model or sequence of operations whose Lipschitz constant should be computed.
        p: The norm to be used for the Lipschitz constant computation. Should be one of [1, 2, np.inf].

    Returns:
        A `LipschitzModel` that computes the Lipschitz constant for the given `backward_model`.

    Raises:
        ValueError: If `backward_model` is neither an instance of `BackwardModel` nor `BackwardSequential`.

    Example:
        ```python
        my_lipschitz_model = get_lipschitz_model(my_backward_model, p=2)
        ```
    """
    if not (
        isinstance(backward_model, BackwardModel) or isinstance(backward_model, BackwardSequential)
    ):
        raise ValueError("the model is not a backward model")

    inputs = backward_model.input
    outputs = backward_model.output

    if isinstance(outputs, list):
        lip_output = [Lipschitz(p)(output_i) for output_i in outputs]
    else:
        lip_output = Lipschitz(p)(outputs)

    return LipschitzModel(
        inputs=inputs,
        outputs=lip_output,
        lipschitz_norm=p,
        backward_model=backward_model,
    )
