"""Squashing activation, gradient formula, and shape flow for the ``tanh`` operator."""

import pytest
import torch

from hndl import resolve
from hndl.operators.tanh import Tanh
from hndl.torch import build


@pytest.mark.parametrize("shape", [(4, 8), (2, 5, 10), (2, 3, 6, 6)])
def test_tanh_parity_and_gradients(device, shape):
    module = Tanh().to(device)
    x = torch.randn(*shape, device=device, requires_grad=True)

    actual = module(x)
    expected = torch.tanh(x)
    assert actual.shape == shape
    assert actual.min() > -1.0 and actual.max() < 1.0
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gx, = torch.autograd.grad(actual, (x,), grad_outputs=grad_out)
    exp_gx = grad_out * (1.0 - expected.square())
    torch.testing.assert_close(gx, exp_gx)


def test_tanh_built_plan(device):
    plan = resolve("conv(3, kernel_size=3, padding=1)\ntanh()",
                   input_shape=("B", 8, 16, 16), output_shape=("B", 3, 16, 16))
    model = build(plan, device=device)
    x = torch.randn(2, 8, 16, 16, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 3, 16, 16)


def test_tanh_bidirectional_shape_flow():
    forward = resolve("tanh()\nlinear(5)", input_shape=("B", 12), output_shape=("B", 5))
    assert forward.nodes[0].output_shapes["out"] == ("B", 12)
    backward = resolve("linear()\ntanh()", input_shape=("B", 6), output_shape=("B", 18))
    assert backward.nodes[0].args["out_features"] == 18
