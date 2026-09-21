"""Numerics, inference, and argument errors for ``hardswish``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import build

SHAPES = [("B", 16), ("B", 4, 8), ("B", 2, 5, 5)]


def pair(shape, device):
    x = torch.randn(3, *shape[1:], device=device) * 3
    return x.clone().requires_grad_(), x.clone().requires_grad_()


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch_functional_forward_and_backward(shape, device):
    plan = resolve("hardswish()", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x, mirror = pair(shape, device)
    actual = model(x=x)["output"]
    expected = F.hardswish(mirror)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_piecewise_branches(device):
    plan = resolve("hardswish()", input_shape=("B", 6), output_shape=("B", 6))
    model = build(plan, device=device)
    x = torch.tensor([[-9.0, -3.0, -1.5, 0.0, 3.0, 9.0]], device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, x * torch.clamp(x + 3, min=0, max=6) / 6)
    assert out[0, 0].item() == 0.0 and out[0, 1].item() == 0.0
    torch.testing.assert_close(out[0, 4:], x[0, 4:])
    assert out[0, 2].item() == pytest.approx(-0.375, rel=1e-6)


def test_shapes_flow_backward_through_the_activation():
    plan = resolve("linear()\nhardswish()", input_shape=("B", 32), output_shape=("B", 7))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7)
    assert plan.nodes[1].input_shapes["x"] == ("B", 7)


def test_shapes_flow_forward_through_the_activation():
    plan = resolve("hardswish()\nlinear()", input_shape=("B", 3, 6), output_shape=("B", 3, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6)


@pytest.mark.parametrize("source", ["hardswish(0.5)", "hardswish(inplace=True)"])
def test_unexpected_arguments_report_e_argument(source):
    with pytest.raises(HNDLError) as excinfo:
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"
