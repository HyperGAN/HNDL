"""Numerics and inference for the ``quick_gelu`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

SHAPES = [("B", 17), ("B", 5, 9), ("B", 3, 6, 7)]
COEFFICIENT = 1.702


def activation(shape, source="quick_gelu()", device="cpu"):
    plan = resolve(source, input_shape=shape, output_shape=shape)
    return build(plan, device=device)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"rank{len(shape)}")
def test_matches_the_sigmoid_definition_on_every_rank(shape, device):
    model = activation(shape, device=device)
    x = torch.randn(4, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    out = model(x=x)["output"]
    expected = mirror * torch.sigmoid(COEFFICIENT * mirror)
    torch.testing.assert_close(out, expected)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_matches_the_closed_form_in_double_precision(device):
    model = activation(("B", 64), device=device)
    x = torch.randn(8, 64, device=device) * 3
    wide = x.double()
    expected = wide / (1.0 + torch.exp(-COEFFICIENT * wide))
    torch.testing.assert_close(model(x=x)["output"].double(), expected, rtol=1e-6, atol=1e-6)


def test_tracks_gelu_closely_without_being_equal(device):
    x = torch.randn(8, 128, device=device) * 2
    quick = activation(("B", 128), device=device)(x=x)["output"]
    exact = activation(("B", 128), "gelu()", device)(x=x)["output"]
    assert not torch.equal(quick, exact)
    torch.testing.assert_close(quick, exact, rtol=0, atol=3e-2)


def test_is_saturating_and_finite_at_the_extremes(device):
    model = activation(("B", 5), device=device)
    x = torch.tensor([[-40.0, -1.0, 0.0, 1.0, 40.0]], device=device)
    out = model(x=x)["output"]
    assert out.isfinite().all()
    torch.testing.assert_close(out[0, 2], torch.zeros((), device=device))
    assert out[0, 0].abs() < 1e-6
    torch.testing.assert_close(out[0, 4], torch.tensor(40.0, device=device))


def test_takes_no_arguments_and_shapes_flow_both_ways():
    with pytest.raises(HNDLError) as error:
        resolve("quick_gelu(0.5)", input_shape=("B", 4), output_shape=("B", 4))
    assert error.value.code in ("E_ARGUMENT", "E_SYNTAX", "E_CONFIG")
    backward = resolve("linear()\nquick_gelu()", input_shape=("B", 5), output_shape=("B", 9))
    assert backward.nodes[0].args["out_features"] == 9
    forward = resolve("quick_gelu()\nlinear()", input_shape=("B", 6, 11), output_shape=("B", 6, 4))
    assert forward.nodes[1].args["in_features"] == 11
    images = resolve("conv(8, kernel_size=3, padding=1)\nquick_gelu()", input_shape=("B", 3, 8, 8),
                     output_shape=("B", 8, 8, 8))
    assert images.nodes[1].output_shapes["out"] == ("B", 8, 8, 8)
