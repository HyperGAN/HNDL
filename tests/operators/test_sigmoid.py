"""Numerics and inference for the ``sigmoid`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import resolve
from hndl.torch import build

SHAPES = [("B", 17), ("B", 5, 9), ("B", 3, 6, 7)]


def activation(shape, source="sigmoid()", device="cpu"):
    plan = resolve(source, input_shape=shape, output_shape=shape)
    return build(plan, device=device)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"rank{len(shape)}")
def test_matches_functional_sigmoid_on_every_rank(shape, device):
    model = activation(shape, device=device)
    x = torch.randn(4, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    out = model(x=x)["output"]
    expected = F.sigmoid(mirror)
    torch.testing.assert_close(out, expected)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_matches_the_closed_form_in_double_precision(device):
    model = activation(("B", 64), device=device)
    x = torch.randn(8, 64, device=device) * 3
    wide = x.double()
    torch.testing.assert_close(model(x=x)["output"].double(), 1.0 / (1.0 + torch.exp(-wide)),
                               rtol=1e-6, atol=1e-6)


def test_output_stays_inside_the_unit_interval_even_at_the_extremes(device):
    model = activation(("B", 5), device=device)
    x = torch.tensor([[-1e4, -30.0, 0.0, 30.0, 1e4]], device=device)
    out = model(x=x)["output"]
    assert out.isfinite().all() and (out >= 0).all() and (out <= 1).all()
    torch.testing.assert_close(out[0, 2], torch.tensor(0.5, device=device))
    assert out[0, 1] < 1e-12 and out[0, 3] > 1 - 1e-6


def test_shapes_flow_forward_and_backward_through_the_activation():
    backward = resolve("linear()\nsigmoid()", input_shape=("B", 5), output_shape=("B", 9))
    assert backward.nodes[0].args["out_features"] == 9
    assert backward.nodes[1].input_shapes["x"] == ("B", 9)
    forward = resolve("sigmoid()\nlinear()", input_shape=("B", 6, 11), output_shape=("B", 6, 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 6, 11)
    assert forward.nodes[1].args["in_features"] == 11
    images = resolve("conv(3, kernel_size=3, padding=1)\nsigmoid()", input_shape=("B", 8, 16, 16),
                     output_shape=("B", 3, 16, 16))
    assert images.nodes[1].output_shapes["out"] == ("B", 3, 16, 16)
