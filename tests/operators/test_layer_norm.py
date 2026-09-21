"""Numerics, inference and error codes specific to ``layer_norm``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build, parameter_counts


def _layer_norm(plan, device):
    """Build the plan and hand back the model together with its norm module."""
    model = build(plan, device=device, initialization_seed=11)
    return model, model[plan.nodes[0].id]


@pytest.mark.parametrize("shape", [(4, 16), (3, 5, 24)])
def test_matches_functional_layer_norm_on_ranks_two_and_three(shape, device):
    plan = resolve("layer_norm()", input_shape=("B", *shape[1:]), output_shape=("B", *shape[1:]))
    model, module = _layer_norm(plan, device)
    with torch.no_grad():
        module.weight.normal_()
        module.bias.normal_()
    x = torch.randn(*shape, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.layer_norm(mirror, (shape[-1],), module.weight, module.bias, module.eps)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_normalizes_the_last_axis_only(device):
    plan = resolve("layer_norm(affine=False)", input_shape=("B", 7, 12), output_shape=("B", 7, 12))
    model, _ = _layer_norm(plan, device)
    out = model(x=torch.randn(3, 7, 12, device=device) * 5 + 2)["output"]
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(3, 7, device=device), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(dim=-1, unbiased=False), torch.ones(3, 7, device=device), atol=1e-3, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype,tolerance", [("float32", 1e-5), ("bfloat16", 3e-2)])
def test_reduced_precision_tracks_the_float32_reference_on_cuda(dtype, tolerance):
    plan = resolve("layer_norm()", input_shape=("B", 6, 32), output_shape=("B", 6, 32), dtype=dtype)
    model, module = _layer_norm(plan, "cuda:0")
    torch_dtype = DTYPES[dtype]
    with torch.no_grad():
        module.weight.normal_()
        module.bias.normal_()
    x = torch.randn(4, 6, 32, device="cuda:0", dtype=torch_dtype)
    actual = model(x=x)["output"]
    assert actual.dtype is torch_dtype
    expected = F.layer_norm(x.float(), (32,), module.weight.float(), module.bias.float(), module.eps)
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("eps", [0.0, -1e-5])
def test_non_positive_eps_is_an_argument_error(eps):
    with pytest.raises(HNDLError, match="E_ARGUMENT.*eps"):
        resolve(f"layer_norm({eps})", input_shape=("B", 8), output_shape=("B", 8))


def test_affine_false_has_no_parameters(device):
    plan = resolve("layer_norm(affine=False)", input_shape=("B", 8), output_shape=("B", 8))
    assert parameter_counts(plan) == {plan.nodes[0].id: 0}
    _, module = _layer_norm(plan, device)
    assert module.weight is None and module.bias is None
    assert list(module.parameters()) == []


def test_construction_settings_initialize_the_weight(device):
    plan = resolve('layer_norm(init={"weight": 0, "bias": 0.5})', input_shape=("B", 8), output_shape=("B", 8))
    assert plan.nodes[0].initialization["overrides"] == {"weight": 0.0, "bias": 0.5}
    model, module = _layer_norm(plan, device)
    torch.testing.assert_close(module.weight, torch.zeros(8, device=device))
    torch.testing.assert_close(module.bias, torch.full((8,), 0.5, device=device))
    out = model(x=torch.randn(2, 8, device=device))["output"]
    torch.testing.assert_close(out, torch.full((2, 8), 0.5, device=device))


def test_width_flows_forward_and_backward_through_the_norm():
    forward = resolve("layer_norm()\nlinear()", input_shape=("B", 16), output_shape=("B", 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 16)
    assert forward.nodes[1].args["in_features"] == 16
    backward = resolve("linear()\nlayer_norm()", input_shape=("B", 8), output_shape=("B", 32))
    assert backward.nodes[0].args["out_features"] == 32
    sequence = resolve("linear(8)\nlayer_norm()\nlinear(3)", input_shape=("B", 5, 8), output_shape=("B", 5, 3))
    assert sequence.nodes[1].output_shapes["out"] == ("B", 5, 8)


def test_a_contradicting_width_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("layer_norm()", input_shape=("B", 8), output_shape=("B", 12))
