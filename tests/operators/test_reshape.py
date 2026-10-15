"""Product dimension preservation, rank changes, and backward solving for ``reshape``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.reshape import Reshape
from hndl.torch import build


@pytest.mark.parametrize("in_shape,target_shape", [
    ((2, 16), (8, 2)),
    ((2, 8192), (32, 16, 16)),
    ((2, 4, 4, 4), (16, 4)),
    ((2, 32), (2, 4, 4)),
])
def test_reshape_parity_and_gradients(device, in_shape, target_shape):
    module = Reshape(shape=target_shape).to(device)
    x = torch.randn(*in_shape, device=device, requires_grad=True)

    actual = module(x)
    expected = x.reshape(in_shape[0], *target_shape)
    assert actual.shape == (in_shape[0], *target_shape)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gx, = torch.autograd.grad(actual, (x,), grad_outputs=grad_out)
    torch.testing.assert_close(gx, grad_out.reshape(in_shape))


def test_reshape_extra_repr():
    module = Reshape(shape=(8, 4, 4))
    assert module.extra_repr() == "shape=(8, 4, 4)"


def test_reshape_built_plan(device):
    plan = resolve("linear()\nreshape(8, 4, 4)", input_shape=("B", 16), output_shape=("B", 8, 4, 4))
    assert plan.nodes[0].args["out_features"] == 128
    model = build(plan, device=device)
    x = torch.randn(2, 16, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 4, 4)


def test_reshape_incompatible_dimensions():
    with pytest.raises(HNDLError, match="E_CONSTRAINT|E_RESHAPE"):
        resolve("reshape(8, 4, 5)", input_shape=("B", 128), output_shape=("B", 8, 4, 5))
