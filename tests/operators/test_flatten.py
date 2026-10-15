"""Numerics, rank collapse, and backward shape inference for the ``flatten`` operator."""

import pytest
import torch

from hndl import resolve
from hndl.operators.flatten import Flatten
from hndl.torch import build


@pytest.mark.parametrize("shape,expected_dim", [
    ((2, 3, 8, 8), 192),
    ((3, 16, 64), 1024),
    ((4, 10), 10),
    ((2, 1, 1, 1), 1),
])
def test_flatten_parity_and_gradients(device, shape, expected_dim):
    module = Flatten().to(device)
    x = torch.randn(*shape, device=device, requires_grad=True)

    actual = module(x)
    expected = torch.flatten(x, start_dim=1)
    assert actual.shape == (shape[0], expected_dim)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gx, = torch.autograd.grad(actual, (x,), grad_outputs=grad_out)
    torch.testing.assert_close(gx, grad_out.reshape(shape))


def test_flatten_built_plan(device):
    plan = resolve("flatten()\nlinear(10)", input_shape=("B", 3, 4, 4), output_shape=("B", 10))
    model = build(plan, device=device)
    x = torch.randn(2, 3, 4, 4, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 10)


def test_flatten_backward_inference():
    plan = resolve("flatten()\nlinear()", input_shape=("B", 2, 4, 4), output_shape=("B", 10))
    assert plan.nodes[1].args["in_features"] == 32
    assert plan.nodes[1].args["out_features"] == 10
