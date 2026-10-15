"""Partitioning, gradient accumulation from multiple branches, and dimension solver for ``split``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.split import Split
from hndl.torch import build


@pytest.mark.parametrize("dim,in_shape,size", [
    (1, (2, 64), 24),
    (1, (2, 128, 16), 48),
    (2, (2, 16, 32), 10),
])
def test_split_parity_and_gradients(device, dim, in_shape, size):
    module = Split(size=size, dim=dim, input_shapes={"x": in_shape}).to(device)
    x = torch.randn(*in_shape, device=device, requires_grad=True)

    first, rest = module(x)
    exp_first, exp_rest = torch.split(x, (size, in_shape[dim] - size), dim=dim)
    torch.testing.assert_close(first, exp_first)
    torch.testing.assert_close(rest, exp_rest)

    g_first = torch.randn_like(first)
    g_rest = torch.randn_like(rest)
    gx, = torch.autograd.grad((first, rest), (x,), grad_outputs=(g_first, g_rest))
    exp_gx = torch.cat((g_first, g_rest), dim=dim)
    torch.testing.assert_close(gx, exp_gx)


def test_split_extra_repr():
    module = Split(size=20, dim=1, input_shapes={"x": (2, 50)})
    assert module.extra_repr() == "sizes=(20, 30), dim=1"


def test_split_built_plan(device):
    source = (
        "z1, z2 = split(64)\n"
        "linear(z1, 32)\n"
        "features = relu()\n"
        "style = linear(z2, 32)\n"
        "add(features, style)"
    )
    plan = resolve(source, input_shape=("B", 128), output_shape=("B", 32))
    model = build(plan, device=device)
    x = torch.randn(2, 128, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 32)


def test_split_size_out_of_bounds():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("a, b = split(150)", input_shape=("B", 100), output_shape={"a": ("B", 150), "b": ("B", 50)})
