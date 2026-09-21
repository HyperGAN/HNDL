"""Numerics, inference, and error codes for ``scale``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

SHAPES = [
    pytest.param(("B", 6), id="rank2"),
    pytest.param(("B", 5, 8), id="rank3"),
    pytest.param(("B", 4, 6, 6), id="rank4"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("factor", [0.5, -2.0, 0.0])
def test_matches_torch_on_every_supported_rank(shape, factor, device):
    plan = resolve(f"scale({factor})", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x = torch.randn(3, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    reference = mirror * factor
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_scale_has_no_parameters_and_preserves_the_shape(device):
    plan = resolve("scale(3.0)", input_shape=("B", 4, 6, 6), output_shape=("B", 4, 6, 6))
    model = build(plan, device=device)
    assert list(model["n0"].parameters()) == [] and list(model["n0"].buffers()) == []
    assert "factor=3.0" in repr(model["n0"])
    x = torch.randn(2, 4, 6, 6, device=device)
    assert model(x=x)["output"].shape == x.shape


def test_scaled_residual_matches_the_handwritten_equivalent(device):
    source = 'saved = x\nlinear(8, name="branch")\nrelu()\nh = linear(4)\ns = scale(h, 0.5)\nadd(s, saved)'
    plan = resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    assert plan.nodes[3].args["factor"] == 0.5
    model = build(plan, device=device, initialization_seed=5)
    x = torch.randn(2, 4, device=device)
    expected = 0.5 * model["n2"](torch.relu(model["branch"](x))) + x
    torch.testing.assert_close(model(x=x)["output"], expected)


def test_shapes_flow_through_scale_in_both_directions():
    backward = resolve("linear()\nscale(0.5)", input_shape=("B", 3), output_shape=("B", 7))
    assert backward.nodes[0].args["out_features"] == 7
    assert backward.nodes[1].input_shapes["x"] == ("B", 7)
    forward = resolve("linear(9)\nscale(0.5)\nlinear()", input_shape=("B", 3), output_shape=("B", 2))
    assert forward.nodes[1].output_shapes["out"] == ("B", 9)
    assert forward.nodes[2].args["in_features"] == 9


def test_an_integer_factor_is_stored_as_a_float_and_round_trips():
    plan = resolve("scale(2)", input_shape=("B", 4), output_shape=("B", 4))
    factor = plan.nodes[0].args["factor"]
    assert type(factor) is float and factor == 2.0


@pytest.mark.parametrize("source,code", [
    ("scale()", "E_ARGUMENT"),
    ("scale('half')", "E_ARGUMENT"),
    ("scale(True)", "E_ARGUMENT"),
    ("scale(0.5, 1.5)", "E_ARGUMENT"),
])
def test_invalid_factors_report_their_code(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 4), output_shape=("B", 4))
