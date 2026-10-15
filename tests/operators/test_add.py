"""Numerics, shape propagation, and branching plans for the ``add`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.add import Add
from hndl.torch import build


@pytest.mark.parametrize("shape", [(4, 8), (2, 5, 10), (2, 3, 8, 8)])
def test_add_parity_and_gradients(device, shape):
    module = Add().to(device)
    a = torch.randn(*shape, device=device, requires_grad=True)
    b = torch.randn(*shape, device=device, requires_grad=True)

    actual = module(a, b)
    expected = torch.add(a, b)
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    ga, gb = torch.autograd.grad(actual, (a, b), grad_outputs=grad_out)
    torch.testing.assert_close(ga, grad_out)
    torch.testing.assert_close(gb, grad_out)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_add_dtypes(device, dtype):
    module = Add().to(device)
    a = torch.randn(2, 4, device=device, dtype=dtype)
    b = torch.randn(2, 4, device=device, dtype=dtype)
    out = module(a, b)
    assert out.dtype == dtype
    torch.testing.assert_close(out, a + b)


def test_add_residual_plan(device):
    source = (
        "saved = x\n"
        "linear(8)\n"
        "relu()\n"
        "h = linear(4)\n"
        "add(h, saved)"
    )
    plan = resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device=device)
    x = torch.randn(3, 4, device=device)
    out = model(x=x)["output"]
    assert out.shape == (3, 4)


def test_add_shape_mismatch_raises_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("add(a, b)", input_shape={"a": ("B", 4), "b": ("B", 8)}, output_shape=("B", 4))
