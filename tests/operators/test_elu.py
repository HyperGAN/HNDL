"""Numerics, inference, and argument errors for ``elu``."""

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
@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
def test_matches_torch_functional_forward_and_backward(shape, alpha, device):
    plan = resolve(f"elu({alpha})", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x, mirror = pair(shape, device)
    actual = model(x=x)["output"]
    expected = F.elu(mirror, alpha=alpha)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_saturates_at_negative_alpha_and_is_identity_above_zero(device):
    plan = resolve("elu(3.0)", input_shape=("B", 5), output_shape=("B", 5))
    model = build(plan, device=device)
    x = torch.tensor([[-100.0, -1.0, 0.0, 1.0, 40.0]], device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out[0, 0], torch.tensor(-3.0, device=device))
    torch.testing.assert_close(out[0, 2:], x[0, 2:])
    assert out[0, 1].item() == pytest.approx(3.0 * (torch.tensor(-1.0).exp().item() - 1.0), rel=1e-6)


def test_shapes_flow_backward_through_the_activation():
    plan = resolve("linear()\nelu()", input_shape=("B", 32), output_shape=("B", 7))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7)
    assert plan.nodes[1].input_shapes["x"] == ("B", 7)


def test_shapes_flow_forward_through_the_activation():
    plan = resolve("elu()\nlinear()", input_shape=("B", 3, 6), output_shape=("B", 3, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6)


@pytest.mark.parametrize("source", ['elu("wide")', "elu(alpha=1.0, inplace=True)"])
def test_bad_arguments_report_e_argument(source):
    with pytest.raises(HNDLError) as excinfo:
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"
