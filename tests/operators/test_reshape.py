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


def test_reshape_negative_dimension(device):
    plan = resolve("reshape(8, -1)", input_shape=("B", 128), output_shape=("B", 8, 16))
    assert plan.nodes[0].args["shape"] == (8, 16)
    model = build(plan, device=device)
    x = torch.randn(2, 128, device=device)
    assert model(x=x)["output"].shape == (2, 8, 16)

    plan2 = resolve("reshape(-1, 8)", input_shape=("B", 128), output_shape=("B", 16, 8))
    assert plan2.nodes[0].args["shape"] == (16, 8)

    plan3 = resolve("reshape(-1)", input_shape=("B", 2, 4, 4), output_shape=("B", 32))
    assert plan3.nodes[0].args["shape"] == (32,)

    # Rank 5 with negative dimension
    plan4 = resolve("reshape(2, 3, -1, 5)", input_shape=("B", 120), output_shape=("B", 2, 3, 4, 5))
    assert plan4.nodes[0].args["shape"] == (2, 3, 4, 5)
    model4 = build(plan4, device=device)
    x4 = torch.randn(2, 120, device=device)
    assert model4(x=x4)["output"].shape == (2, 2, 3, 4, 5)


def test_reshape_negative_dimension_validation_errors():
    with pytest.raises(HNDLError, match="At most one -1 dimension permitted"):
        resolve("reshape(-1, -1)", input_shape=("B", 64), output_shape=("B", 8, 8))

    with pytest.raises(HNDLError, match="Dimensions must be positive or -1"):
        resolve("reshape(0, 16)", input_shape=("B", 64), output_shape=("B", 4, 16))

    with pytest.raises(HNDLError, match="shape must be >= -1"):
        resolve("reshape(-2, 16)", input_shape=("B", 64), output_shape=("B", 4, 16))

    with pytest.raises(HNDLError, match="at most four dimensions"):
        resolve("reshape(2, 2, 2, 2, 2)", input_shape=("B", 32), output_shape=("B", 2, 2, 2, 2, 2))
