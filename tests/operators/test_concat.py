"""Numerics, axes, batch joins, and shape inference for the ``concat`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.concat import Concat
from hndl.torch import build


@pytest.mark.parametrize("axis", [1, 2])
def test_concat_parity_and_gradients(device, axis):
    module = Concat(input_count=2, axis=axis).to(device)
    s1 = [2, 4, 6]
    s2 = [2, 4, 6]
    s1[axis] = 3
    s2[axis] = 5
    a = torch.randn(*s1, device=device, requires_grad=True)
    b = torch.randn(*s2, device=device, requires_grad=True)

    actual = module(a, b)
    expected = torch.cat((a, b), dim=axis)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    ga, gb = torch.autograd.grad(actual, (a, b), grad_outputs=grad_out)
    exp_ga, exp_gb = torch.split(grad_out, [s1[axis], s2[axis]], dim=axis)
    torch.testing.assert_close(ga, exp_ga)
    torch.testing.assert_close(gb, exp_gb)


def test_concat_batch_axis_join(device):
    source = (
        "a = linear(4)\n"
        "b = linear(x, 4)\n"
        "pair = concat(a, b, axis=0)\n"
        "shared = linear(pair, 6)\n"
        "p, q = chunk(shared, 2, dim=0)\n"
        "concat(p, q)"
    )
    plan = resolve(source, input_shape=("B", 8), output_shape=("B", 12))
    model = build(plan, device=device)
    x = torch.randn(2, 8, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 12)


def test_concat_extra_repr():
    cat = Concat(input_count=3, axis=1)
    assert cat.extra_repr() == "input_count=3, axis=1"


def test_concat_axis_out_of_bounds():
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("concat(a, b, axis=3)",
                input_shape={"a": ("B", 4), "b": ("B", 4)},
                output_shape=("B", 8))
