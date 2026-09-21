"""Numerics, inference, and argument errors for the ``gelu`` operator."""

import math

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import build

SHAPES = [("B", 17), ("B", 5, 9), ("B", 3, 6, 7)]


def activation(shape, source="gelu()", device="cpu"):
    plan = resolve(source, input_shape=shape, output_shape=shape)
    return build(plan, device=device)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"rank{len(shape)}")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_matches_functional_gelu_on_every_rank(shape, approximate, device):
    model = activation(shape, f'gelu(approximate="{approximate}")', device)
    x = torch.randn(4, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    out = model(x=x)["output"]
    expected = F.gelu(mirror, approximate=approximate)
    torch.testing.assert_close(out, expected)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_exact_form_matches_the_erf_definition_in_double_precision(device):
    model = activation(("B", 64), "gelu()", device)
    x = torch.randn(8, 64, device=device) * 3
    out = model(x=x)["output"].double()
    wide = x.double()
    expected = 0.5 * wide * (1 + torch.erf(wide / math.sqrt(2.0)))
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_tanh_form_matches_its_definition_and_differs_from_the_exact_form(device):
    x = torch.randn(8, 64, device=device) * 3
    exact = activation(("B", 64), 'gelu(approximate="none")', device)(x=x)["output"]
    tanh = activation(("B", 64), 'gelu(approximate="tanh")', device)(x=x)["output"]
    wide = x.double()
    definition = 0.5 * wide * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (wide + 0.044715 * wide.pow(3))))
    torch.testing.assert_close(tanh.double(), definition, rtol=1e-6, atol=1e-6)
    assert not torch.equal(exact, tanh)
    assert (exact - tanh).abs().max() > 1e-5
    torch.testing.assert_close(exact, tanh, rtol=0, atol=2e-3)


def test_default_is_the_exact_form_and_the_choice_is_recorded_in_the_plan():
    plan = resolve("gelu()", input_shape=("B", 4), output_shape=("B", 4))
    assert plan.nodes[0].args["approximate"] == "none"
    other = resolve('gelu(approximate="tanh")', input_shape=("B", 4), output_shape=("B", 4))
    assert other.nodes[0].args["approximate"] == "tanh"
    assert plan.semantic_digest != other.semantic_digest


def test_shapes_flow_forward_and_backward_through_the_activation():
    backward = resolve("linear()\ngelu()", input_shape=("B", 5), output_shape=("B", 9))
    assert backward.nodes[0].args["out_features"] == 9
    assert backward.nodes[1].input_shapes["x"] == ("B", 9)
    forward = resolve("gelu()\nlinear()", input_shape=("B", 6, 11), output_shape=("B", 6, 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 6, 11)
    assert forward.nodes[1].args["in_features"] == 11
    images = resolve("conv(8, kernel_size=3, padding=1)\ngelu()", input_shape=("B", 3, 8, 8),
                     output_shape=("B", 8, 8, 8))
    assert images.nodes[1].output_shapes["out"] == ("B", 8, 8, 8)


@pytest.mark.parametrize("source", ['gelu(approximate="bogus")', 'gelu(approximate="")', "gelu(approximate=3)"])
def test_invalid_approximate_is_an_argument_error(source):
    with pytest.raises(HNDLError) as error:
        resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    assert error.value.code == "E_ARGUMENT"
