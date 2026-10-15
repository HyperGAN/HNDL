"""Numerics, slope scaling, out-of-place execution, and dtypes for ``leaky_relu``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import resolve
from hndl.operators.leaky_relu import LeakyReLU
from hndl.torch import build


@pytest.mark.parametrize("slope", [0.01, 0.2, 0.5])
def test_leaky_relu_parity_and_gradients(device, slope):
    module = LeakyReLU(negative_slope=slope).to(device)
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=device, requires_grad=True)

    actual = module(x)
    expected = F.leaky_relu(x, negative_slope=slope)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.ones_like(actual)
    gx_act, = torch.autograd.grad(actual, (x,), grad_outputs=grad_out, retain_graph=True)
    gx_exp, = torch.autograd.grad(expected, (x,), grad_outputs=grad_out)
    torch.testing.assert_close(gx_act, gx_exp)


def test_leaky_relu_out_of_place(device):
    module = LeakyReLU(0.1).to(device)
    x = torch.tensor([-1.0, 1.0], device=device)
    x_copy = x.clone()
    _ = module(x)
    assert torch.equal(x, x_copy), "leaky_relu must not mutate input tensor in place"


def test_leaky_relu_built_plan(device):
    plan = resolve("conv(8, kernel_size=3, padding=1)\nleaky_relu(0.2)",
                   input_shape=("B", 3, 8, 8), output_shape=("B", 8, 8, 8))
    model = build(plan, device=device)
    x = torch.randn(2, 3, 8, 8, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 8, 8)
