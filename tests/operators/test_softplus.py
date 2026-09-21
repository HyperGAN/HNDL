"""Numerics, inference, and argument errors for ``softplus``."""

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
@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (2.0, 20.0), (0.5, 4.0)])
def test_matches_torch_functional_forward_and_backward(shape, beta, threshold, device):
    plan = resolve(f"softplus({beta}, threshold={threshold})", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x, mirror = pair(shape, device)
    actual = model(x=x)["output"]
    expected = F.softplus(mirror, beta=beta, threshold=threshold)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_output_is_positive_and_the_threshold_branch_is_the_identity(device):
    plan = resolve("softplus()", input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device=device)
    x = torch.tensor([[-60.0, 0.0, 3.0, 400.0]], device=device)
    out = model(x=x)["output"]
    assert (out >= 0).all() and out.isfinite().all()
    assert out[0, 1].item() == pytest.approx(0.6931472, rel=1e-6)
    torch.testing.assert_close(out[0, 3], x[0, 3])


def test_large_beta_approaches_relu(device):
    plan = resolve("softplus(8.0)", input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device=device)
    x = torch.tensor([[-4.0, -2.0, 2.0, 4.0]], device=device)
    torch.testing.assert_close(model(x=x)["output"], F.relu(x), atol=1e-3, rtol=0)


def test_shapes_flow_backward_through_the_activation():
    plan = resolve("linear()\nsoftplus()", input_shape=("B", 32), output_shape=("B", 7))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7)
    assert plan.nodes[1].input_shapes["x"] == ("B", 7)


def test_shapes_flow_forward_through_the_activation():
    plan = resolve("softplus()\nlinear()", input_shape=("B", 3, 6), output_shape=("B", 3, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6)


@pytest.mark.parametrize("source", ["softplus(0.0)", "softplus(-1.0)", "softplus(1.0, threshold=0.0)",
                                    "softplus(sharpness=2.0)"])
def test_bad_arguments_report_e_argument(source):
    with pytest.raises(HNDLError) as excinfo:
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"
