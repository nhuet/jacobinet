# test the implementation of FGSM and PGD against other frameworks: aka torch attacks
import keras.ops as K
import numpy as np
import pytest
import torch
import torch.nn as nn
import torchattacks
from jacobinet.attacks import get_adv_model
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model, Sequential

from .conftest import get_toy_model_functional, get_toy_model_sequential


def _test_fgsm_logits(model, eps, input_shape_wo_batch, output_shape_wo_batch, norm=np.inf):
    # build FGSM model
    fgsm_model = get_adv_model(model, loss="logits", epsilon=eps, attack="fgsm", p=norm)

    random_input = np.asarray(np.random.rand(*(1,) + input_shape_wo_batch), dtype="float32")
    y_pred = model.predict(random_input).argmax(-1)
    one_hot = np.zeros(output_shape_wo_batch)
    one_hot[y_pred] = 1
    # generate an attack

    k_adv_input = fgsm_model.predict([random_input, one_hot[None]])

    # generate attack by hand
    input_torch = torch.tensor(random_input, requires_grad=True)
    output = model(input_torch)
    select_output = output[0, y_pred[0]]
    select_output.backward()
    gradient = input_torch.grad.cpu().detach().numpy()[0]

    if norm == np.inf:
        np_attack = random_input[0] + eps * np.sign(gradient)
    elif norm == 2:
        np_attack = random_input[0] + eps * gradient / np.sqrt(np.sum(gradient**2))
    np.testing.assert_almost_equal(k_adv_input[0], np_attack, decimal=3)


def _test_fgsm_crossentropy(model, eps, input_shape_wo_batch, output_shape_wo_batch, norm=np.inf):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    class Attack(nn.Module):
        def __init__(self, model):
            super().__init__()

            self.keras_model = model

        def forward(self, x):
            return self.keras_model(x)

    # embbed the keras model into nn.Module
    torch_model = Attack(model)
    # build FGSM model
    fgsm_model = get_adv_model(
        model, loss="categorical_crossentropy", epsilon=eps, attack="fgsm", p=norm
    )

    random_input = np.asarray(np.random.rand(*(1,) + input_shape_wo_batch), dtype="float32")
    y_pred = model.predict(random_input).argmax(-1)
    one_hot = np.zeros(output_shape_wo_batch, dtype="float32")
    one_hot[y_pred] = 1

    k_adv_input = fgsm_model.predict([random_input, one_hot[None]])

    # compute gradient by hand ...
    t_input = torch.tensor(random_input, requires_grad=True, device=device)
    loss = torch.nn.CrossEntropyLoss()
    output = loss(torch_model(t_input), torch.tensor(one_hot, device=device)[None])
    output.backward()

    gradient = t_input.grad.cpu().detach().numpy()[0]

    if norm == np.inf:
        np_attack = random_input[0] + eps * np.sign(gradient)
    elif norm == 2:
        np_attack = random_input[0] + eps * gradient / np.sqrt(np.sum(gradient**2))

    np.testing.assert_almost_equal(k_adv_input[0], np_attack, decimal=3)

    if norm == np.inf:
        t_attack = torchattacks.FGSM(torch_model, eps=eps)  # infinite norm
        t_adv_input = (
            t_attack(torch.tensor(random_input), torch.tensor(y_pred)).cpu().detach().numpy()
        )
        np.testing.assert_almost_equal(k_adv_input, t_adv_input, decimal=1)


def _test_fgsm_bounds(model, eps, input_shape_wo_batch, output_shape_wo_batch, norm, lower, upper):
    # build FGSM model
    fgsm_model_upper = get_adv_model(
        model, loss="logits", epsilon=upper + 1, attack="fgsm", p=norm, upper=upper, lower=lower
    )

    fgsm_model_lower = get_adv_model(
        model, loss="logits", epsilon=lower + 1, attack="fgsm", p=norm, upper=upper, lower=lower
    )

    random_input = np.asarray(np.random.rand(*(1,) + input_shape_wo_batch), dtype="float32")
    y_pred = model.predict(random_input).argmax(-1)
    one_hot = np.zeros(output_shape_wo_batch)
    one_hot[y_pred] = 1
    # generate an attack

    k_adv_input_u = fgsm_model_upper.predict([random_input, one_hot[None]])
    k_adv_input_l = fgsm_model_upper.predict([random_input, one_hot[None]])

    import pdb

    pdb.set_trace()


def _test_pdg_crossentropy(
    model, eps, alpha, steps, input_shape_wo_batch, output_shape_wo_batch, norm=np.inf
):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    class Attack(nn.Module):
        def __init__(self, model):
            super().__init__()

            self.keras_model = model

        def forward(self, x):
            x = x.view(*(1,) + input_shape_wo_batch)
            return self.keras_model(x)

    # embbed the keras model into nn.Module
    torch_model = Attack(model)
    # build FGSM model
    pgd_model = get_adv_model(
        model,
        loss="categorical_crossentropy",
        epsilon=eps,
        alpha=alpha,
        n_iter=steps,
        attack="pgd",
        p=norm,
        random_init=False,
    )

    random_input = np.asarray(np.random.rand(*(1,) + input_shape_wo_batch), dtype="float32")
    y_pred = model.predict(random_input).argmax(-1)
    one_hot = np.zeros(output_shape_wo_batch, dtype="float32")
    one_hot[y_pred] = 1

    k_adv_input = pgd_model.predict([random_input, one_hot[None]])

    if norm == np.inf:
        t_attack = torchattacks.PGD(
            torch_model, eps=eps, alpha=alpha, steps=steps, random_start=False
        )  # infinite norm
        t_adv_input = (
            t_attack(torch.tensor(random_input), torch.tensor(y_pred)).cpu().detach().numpy()
        )

    else:
        t_attack = torchattacks.PGDL2(
            torch_model, eps=eps, alpha=alpha, steps=steps, random_start=False
        )  # infinite norm

        t_adv_input = (
            t_attack(torch.tensor(random_input), torch.tensor(y_pred)).cpu().detach().numpy()
        )
    # compute mse
    mse = np.mean(np.sum((t_adv_input - k_adv_input) ** 2))

    try:
        np.testing.assert_almost_equal(k_adv_input, t_adv_input, decimal=4)
    except:
        assert mse <= 0.5


@pytest.mark.parametrize(
    "loss, model_type, norm",
    [
        ("logits", "sequential", np.inf),
        ("logits", "functional", np.inf),
        ("categorical_crossentropy", "sequential", np.inf),
        ("categorical_crossentropy", "functional", np.inf),
        ("logits", "sequential", 2),
        ("logits", "functional", 2),
        ("categorical_crossentropy", "sequential", 2),
        ("categorical_crossentropy", "functional", 2),
    ],
)
def test_fgsm(loss, model_type, norm):
    input_shape_wo_batch = (16,)
    output_dim_wo_batch = 10
    if model_type == "sequential":
        model = get_toy_model_sequential(
            input_shape_wo_batch=input_shape_wo_batch, output_dim_wo_batch=output_dim_wo_batch
        )
    elif model_type == "functional":
        model = get_toy_model_functional(
            input_shape_wo_batch=input_shape_wo_batch, output_dim_wo_batch=output_dim_wo_batch
        )
    else:
        raise ValueError("unknown model_type {}".format(model_type))

    if loss == "logits":
        _test_fgsm_logits(
            model,
            eps=0.1,
            input_shape_wo_batch=input_shape_wo_batch,
            output_shape_wo_batch=(output_dim_wo_batch,),
            norm=norm,
        )
    elif loss == "categorical_crossentropy":
        _test_fgsm_crossentropy(
            model,
            eps=0.1,
            input_shape_wo_batch=input_shape_wo_batch,
            output_shape_wo_batch=(output_dim_wo_batch,),
            norm=norm,
        )
    else:
        raise ValueError("unknown loss {}".format(loss))


@pytest.mark.parametrize(
    "model_type, norm, steps",
    [
        ("sequential", np.inf, 1),
        ("functional", np.inf, 1),
        ("sequential", 2, 1),
        ("functional", 2, 1),
        ("sequential", np.inf, 3),
        ("functional", np.inf, 3),
        ("sequential", 2, 3),
        ("functional", 2, 3),
    ],
)
def test_pgd(model_type, norm, steps):
    input_shape_wo_batch = (16,)
    output_dim_wo_batch = 10
    eps = 1
    alpha = 0.1
    if model_type == "sequential":
        model = get_toy_model_sequential(
            input_shape_wo_batch=input_shape_wo_batch, output_dim_wo_batch=output_dim_wo_batch
        )
    elif model_type == "functional":
        model = get_toy_model_functional(
            input_shape_wo_batch=input_shape_wo_batch, output_dim_wo_batch=output_dim_wo_batch
        )
    else:
        raise ValueError("unknown model_type {}".format(model_type))

    _test_pdg_crossentropy(
        model, eps, alpha, steps, input_shape_wo_batch, (output_dim_wo_batch,), norm=norm
    )
