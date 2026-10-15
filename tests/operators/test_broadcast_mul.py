"""Numerics, gradient reductions, and shape broadcasting for the ``broadcast_mul`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.broadcast_mul import BroadcastMul
from hndl.torch import build


@pytest.mark.parametrize("a_shape,b_shape", [
    ((2, 3, 4), (2, 1, 4)),
    ((2, 1, 4), (2, 3, 1)),
    ((2, 4, 8, 8), (2, 4, 1, 1)),
    ((2, 6, 8), (2, 1, 8)),
])
def test_broadcast_mul_parity_and_gradients(device, a_shape, b_shape):
    module = BroadcastMul().to(device)
    a = torch.randn(*a_shape, device=device, requires_grad=True)
    b = torch.randn(*b_shape, device=device, requires_grad=True)

    actual = module(a, b)
    expected = torch.mul(a, b)
    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    assert actual.shape == out_shape
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    ga, gb = torch.autograd.grad(actual, (a, b), grad_outputs=grad_out)
    torch.testing.assert_close(ga, (grad_out * b).sum_to_size(a.shape))
    torch.testing.assert_close(gb, (grad_out * a).sum_to_size(b.shape))


def test_broadcast_mul_built_plan(device):
    source = (
        "h = linear(8)\n"
        "g = mean(h, 1, keepdim=True)\n"
        "broadcast_mul(h, g)"
    )
    plan = resolve(source, input_shape=("B", 6, 4), output_shape=("B", 6, 8))
    model = build(plan, device=device)
    x = torch.randn(2, 6, 4, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 6, 8)


def test_broadcast_mul_backward_shape_inference():
    source = (
        "gate = mean(2, keepdim=True)\n"
        "h = linear(x)\n"
        "broadcast_mul(h, gate)"
    )
    plan = resolve(source, input_shape=("B", 6, 4), output_shape=("B", 6, 8))
    assert plan.nodes[1].args["out_features"] == 8


def test_broadcast_mul_incompatible_shapes():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("broadcast_mul(a, b)",
                input_shape={"a": ("B", 3, 4), "b": ("B", 3, 5)},
                output_shape=("B", 3, 4))
