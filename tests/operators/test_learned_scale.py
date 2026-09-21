"""Numerics, the learned parameter, inference, and error codes for ``learned_scale``."""

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
@pytest.mark.parametrize("init_value", [0.0, 0.5, -2.0])
def test_matches_torch_on_every_supported_rank(shape, init_value, device):
    plan = resolve(f"learned_scale(init_value={init_value})", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    gamma = model["n0"].gamma
    assert torch.equal(gamma.detach(), torch.full((1,), init_value, device=device))
    x = torch.randn(3, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    reference_gamma = torch.full((1,), init_value, device=device, requires_grad=True)
    actual = model(x=x)["output"]
    reference = reference_gamma * mirror
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)
    torch.testing.assert_close(gamma.grad, reference_gamma.grad)


def test_the_default_gate_starts_as_a_zero_map_so_a_residual_is_the_identity(device):
    source = ("saved = x\nconv(4, kernel_size=3, padding=1)\nrelu()\nh = conv(4, kernel_size=1)\n"
              "s = learned_scale(h)\nadd(s, saved)")
    plan = resolve(source, input_shape=("B", 4, 8, 8), output_shape=("B", 4, 8, 8))
    assert plan.nodes[3].args["init_value"] == 0.0
    model = build(plan, device=device, initialization_seed=5)
    x = torch.randn(2, 4, 8, 8, device=device)
    torch.testing.assert_close(model(x=x)["output"], x)


def test_the_single_parameter_is_named_gamma(device):
    plan = resolve("learned_scale(init_value=1.5)", input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device=device)
    assert [name for name, _ in model["n0"].named_parameters()] == ["gamma"]
    assert model["n0"].gamma.shape == (1,) and model["n0"].gamma.numel() == 1
    assert list(model["n0"].buffers()) == []
    assert "init_value=1.5" in repr(model["n0"])


def test_construction_overrides_target_gamma_by_name(device):
    plan = resolve('learned_scale(init_value=0.25, init={"gamma": 0.75}, trainable={"gamma": False})',
                   input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device=device)
    gamma = model["n0"].gamma
    assert torch.equal(gamma.detach(), torch.full((1,), 0.75, device=device))
    assert gamma.requires_grad is False


def test_an_override_on_an_unknown_parameter_is_rejected(device):
    plan = resolve('learned_scale(init={"weight": 1.0})', input_shape=("B", 4), output_shape=("B", 4))
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        build(plan, device=device)


def test_shapes_flow_through_learned_scale_in_both_directions():
    backward = resolve("linear()\nlearned_scale()", input_shape=("B", 3), output_shape=("B", 7))
    assert backward.nodes[0].args["out_features"] == 7
    assert backward.nodes[1].input_shapes["x"] == ("B", 7)
    forward = resolve("linear(9)\nlearned_scale()\nlinear()", input_shape=("B", 3), output_shape=("B", 2))
    assert forward.nodes[1].output_shapes["out"] == ("B", 9)
    assert forward.nodes[2].args["in_features"] == 9


def test_an_integer_init_value_is_stored_as_a_float_and_round_trips():
    plan = resolve("learned_scale(init_value=2)", input_shape=("B", 4), output_shape=("B", 4))
    value = plan.nodes[0].args["init_value"]
    assert type(value) is float and value == 2.0


@pytest.mark.parametrize("source,code", [
    ("learned_scale(0.5)", "E_ARGUMENT"),
    ("learned_scale(init_value='half')", "E_ARGUMENT"),
    ("learned_scale(init_value=True)", "E_ARGUMENT"),
    ("learned_scale(gamma=0.5)", "E_ARGUMENT"),
])
def test_invalid_arguments_report_their_code(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 4), output_shape=("B", 4))
