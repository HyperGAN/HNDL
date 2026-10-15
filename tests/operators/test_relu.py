"""Numerics, out-of-place execution, dtypes, and shape propagation for the ``relu`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import resolve
from hndl.operators.relu import ReLU
from hndl.torch import build


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8), (2, 3, 4, 4)])
def test_relu_parity_and_gradients(device, shape):
    module = ReLU().to(device)
    x = torch.randn(*shape, device=device, requires_grad=True)

    actual = module(x)
    expected = F.relu(x)
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected)

    grad_out = torch.ones_like(actual)
    gx, = torch.autograd.grad(actual, (x,), grad_outputs=grad_out)
    exp_gx = (x > 0).float()
    torch.testing.assert_close(gx, exp_gx)


def test_relu_out_of_place(device):
    module = ReLU().to(device)
    x = torch.tensor([-2.0, 1.0], device=device)
    x_copy = x.clone()
    _ = module(x)
    assert torch.equal(x, x_copy), "relu must not mutate input tensor in place"


def test_relu_built_plan(device):
    plan = resolve("linear(64)\nrelu()\nlinear(10)", input_shape=("B", 128), output_shape=("B", 10))
    model = build(plan, device=device)
    x = torch.randn(2, 128, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 10)


def test_relu_bidirectional_shape_flow():
    forward = resolve("relu()\nlinear(10)", input_shape=("B", 16), output_shape=("B", 10))
    assert forward.nodes[0].output_shapes["out"] == ("B", 16)
    backward = resolve("linear()\nrelu()", input_shape=("B", 8), output_shape=("B", 32))
    assert backward.nodes[0].args["out_features"] == 32
