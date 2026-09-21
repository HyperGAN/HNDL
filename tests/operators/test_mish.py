"""Numerics, inference, and argument errors for ``mish``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import build

SHAPES = [("B", 16), ("B", 4, 8), ("B", 2, 5, 5)]


def pair(shape, device):
    x = torch.randn(3, *shape[1:], device=device)
    return x.clone().requires_grad_(), x.clone().requires_grad_()


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch_functional_forward_and_backward(shape, device):
    plan = resolve("mish()", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x, mirror = pair(shape, device)
    actual = model(x=x)["output"]
    expected = F.mish(mirror)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_matches_the_closed_form_and_stays_finite_at_the_extremes(device):
    plan = resolve("mish()", input_shape=("B", 6), output_shape=("B", 6))
    model = build(plan, device=device)
    x = torch.tensor([[-90.0, -1.0, 0.0, 1.0, 5.0, 90.0]], device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, x * torch.tanh(F.softplus(x)))
    assert out.isfinite().all()
    torch.testing.assert_close(out[0, 2], torch.tensor(0.0, device=device))
    torch.testing.assert_close(out[0, 5], x[0, 5])
    assert out[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_shapes_flow_backward_through_the_activation():
    plan = resolve("linear()\nmish()", input_shape=("B", 32), output_shape=("B", 7))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7)
    assert plan.nodes[1].input_shapes["x"] == ("B", 7)


def test_shapes_flow_forward_through_the_activation():
    plan = resolve("mish()\nlinear()", input_shape=("B", 3, 6), output_shape=("B", 3, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6)


@pytest.mark.parametrize("source", ["mish(1.0)", "mish(inplace=True)"])
def test_unexpected_arguments_report_e_argument(source):
    with pytest.raises(HNDLError) as excinfo:
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"
