"""Numerics and inference for the ``silu`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import resolve
from hndl.torch import build

SHAPES = [("B", 17), ("B", 5, 9), ("B", 3, 6, 7)]


def activation(shape, source="silu()", device="cpu"):
    plan = resolve(source, input_shape=shape, output_shape=shape)
    return build(plan, device=device)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"rank{len(shape)}")
def test_matches_functional_silu_on_every_rank(shape, device):
    model = activation(shape, device=device)
    x = torch.randn(4, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    out = model(x=x)["output"]
    expected = F.silu(mirror)
    torch.testing.assert_close(out, expected)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_matches_the_closed_form_in_double_precision(device):
    model = activation(("B", 64), device=device)
    x = torch.randn(8, 64, device=device) * 3
    wide = x.double()
    torch.testing.assert_close(model(x=x)["output"].double(), wide * torch.sigmoid(wide), rtol=1e-6, atol=1e-6)


def test_has_no_parameters_and_the_documented_minimum(device):
    model = activation(("B", 4001), device=device)
    assert list(model.parameters()) == []
    x = torch.linspace(-4.0, 4.0, 4001, device=device).unsqueeze(0)
    out = model(x=x)["output"]
    value, index = out.min(dim=1)
    torch.testing.assert_close(value[0], torch.tensor(-0.2784645, device=device), rtol=0, atol=1e-5)
    torch.testing.assert_close(x[0, index[0]], torch.tensor(-1.2784, device=device), rtol=0, atol=2e-3)
    assert out[0, -1] > 3.9


def test_shapes_flow_forward_and_backward_through_the_activation():
    backward = resolve("linear()\nsilu()", input_shape=("B", 5), output_shape=("B", 9))
    assert backward.nodes[0].args["out_features"] == 9
    assert backward.nodes[1].input_shapes["x"] == ("B", 9)
    forward = resolve("silu()\nlinear()", input_shape=("B", 6, 11), output_shape=("B", 6, 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 6, 11)
    assert forward.nodes[1].args["in_features"] == 11
    images = resolve("conv(8, kernel_size=3, padding=1)\nsilu()", input_shape=("B", 3, 8, 8),
                     output_shape=("B", 8, 8, 8))
    assert images.nodes[1].output_shapes["out"] == ("B", 8, 8, 8)
