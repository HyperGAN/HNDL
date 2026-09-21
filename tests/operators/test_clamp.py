"""Numerics, inference, and error codes for the ``clamp`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.clamp import Clamp
from hndl.torch import DTYPES, build

from .conftest import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,bounds", [
    ((4, 9), (-0.5, 0.5)),
    ((3, 5, 7), (0.0, 6.0)),
    ((2, 3, 4, 4), (-1.0, 1.0)),
])
def test_matches_torch_clamp_and_its_gradient_on_random_tensors(device, shape, bounds):
    low, high = bounds
    module = Clamp(min=low, max=high).to(device)
    x = torch.randn(*shape, device=device) * 3
    mirror = x.detach().clone().requires_grad_()
    x.requires_grad_()

    actual = module(x)
    expected = torch.clamp(mirror, min=low, max=high)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    assert actual.min() >= low and actual.max() <= high

    weight = torch.randn_like(x)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)
    # Saturated elements stop contributing to the gradient.
    saturated = (x.detach() < low) | (x.detach() > high)
    assert torch.equal(x.grad[saturated], torch.zeros_like(x.grad[saturated]))


@pytest.mark.parametrize("device", DEVICES)
def test_built_plan_saturates_hidden_activations(device):
    plan = resolve("linear(6)\nclamp(0.0, 2.0)\nlinear()", input_shape=("B", 5), output_shape=("B", 3))
    assert plan.nodes[1].args == {"min": 0.0, "max": 2.0}
    assert plan.nodes[1].input_shapes["x"] == ("B", 6) == plan.nodes[1].output_shapes["out"]
    model = build(plan, device=device, initialization_seed=11)
    x = torch.randn(4, 5, device=device)
    hidden = model[plan.nodes[0].id](x)
    torch.testing.assert_close(model[plan.nodes[1].id](hidden), torch.clamp(hidden, min=0.0, max=2.0))


def test_shape_and_rank_flow_through_the_operator_in_both_directions():
    forward = resolve("clamp(-1.0, 1.0)\nlinear()", input_shape=("B", 7, 4), output_shape=("B", 7, 2))
    assert forward.nodes[0].output_shapes["out"] == ("B", 7, 4)
    backward = resolve("linear()\nclamp(0.0, 1.0)", input_shape=("B", 8), output_shape=("B", 3))
    assert backward.nodes[0].args["out_features"] == 3
    assert backward.nodes[1].input_shapes["x"] == ("B", 3)
    image = resolve("clamp(0.0, 1.0)", input_shape=("B", 3, 6, 6), output_shape=("B", 3, 6, 6))
    assert image.nodes[0].output_shapes["out"] == ("B", 3, 6, 6)


def test_degenerate_interval_is_allowed_and_produces_a_constant():
    plan = resolve("clamp(0.5, 0.5)", input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device="cpu")
    output = model(x=torch.randn(3, 4))["output"]
    assert torch.equal(output, torch.full_like(output, 0.5))


@pytest.mark.parametrize("source,message", [
    ("clamp(1.0, -1.0)", "min <= max"),
    ("clamp(0.5, 0.25)", "min <= max"),
    ("clamp(0.0)", "requires max"),
    ("clamp()", "requires min"),
    ('clamp("a", 1.0)', "must be a finite float"),
    ("clamp(0.0, 1.0, 2.0)", "at most 2"),
])
def test_invalid_bounds_report_e_argument(source, message):
    with pytest.raises(HNDLError, match="E_ARGUMENT") as failure:
        resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    assert message in failure.value.message


def test_integer_bounds_are_accepted_and_normalized_to_floats():
    plan = resolve("clamp(0, 1)", input_shape=("B", 4), output_shape=("B", 4))
    args = plan.nodes[0].args
    assert args == {"min": 0.0, "max": 1.0}
    assert all(type(value) is float for value in args.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_clamping_stays_in_the_input_dtype(dtype):
    plan = resolve("clamp(-0.5, 0.5)", input_shape=("B", 16), output_shape=("B", 16), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(4, 16, device="cuda:0", dtype=torch_dtype) * 4
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype
    torch.testing.assert_close(output, torch.clamp(x, min=-0.5, max=0.5))
