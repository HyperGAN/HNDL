"""Numerics, inference and error codes specific to ``rms_norm``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build, parameter_counts


def _rms_norm(plan, device):
    """Build the plan and hand back the model together with its norm module."""
    model = build(plan, device=device, initialization_seed=7)
    return model, model[plan.nodes[0].id]


def _manual(x, weight, eps):
    """out = x / sqrt(mean(x^2)) * weight, written out in double precision."""
    high = x.double()
    out = high * torch.rsqrt(high.square().mean(dim=-1, keepdim=True) + eps)
    return out if weight is None else out * weight.double()


@pytest.mark.parametrize("shape", [(4, 16), (3, 5, 24)])
def test_matches_the_manual_formula_on_ranks_two_and_three(shape, device):
    plan = resolve("rms_norm()", input_shape=("B", *shape[1:]), output_shape=("B", *shape[1:]))
    model, module = _rms_norm(plan, device)
    with torch.no_grad():
        module.weight.normal_()
    x = torch.randn(*shape, device=device)
    torch.testing.assert_close(model(x=x)["output"].double(), _manual(x, module.weight, module.eps),
                               atol=1e-6, rtol=1e-6)


def test_matches_functional_rms_norm_including_gradients(device):
    plan = resolve("rms_norm(affine=False)", input_shape=("B", 6, 20), output_shape=("B", 6, 20))
    model, module = _rms_norm(plan, device)
    x = torch.randn(3, 6, 20, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.rms_norm(mirror, (20,), None, module.eps)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_the_root_mean_square_of_the_output_is_one(device):
    plan = resolve("rms_norm(affine=False)", input_shape=("B", 7, 12), output_shape=("B", 7, 12))
    model, _ = _rms_norm(plan, device)
    out = model(x=torch.randn(3, 7, 12, device=device) * 5 + 2)["output"]
    torch.testing.assert_close(out.square().mean(dim=-1), torch.ones(3, 7, device=device), atol=1e-4, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype,tolerance", [("float32", 1e-5), ("bfloat16", 3e-2)])
def test_reduced_precision_tracks_the_double_precision_reference_on_cuda(dtype, tolerance):
    plan = resolve("rms_norm()", input_shape=("B", 6, 32), output_shape=("B", 6, 32), dtype=dtype)
    model, module = _rms_norm(plan, "cuda:0")
    torch_dtype = DTYPES[dtype]
    with torch.no_grad():
        module.weight.normal_()
    x = torch.randn(4, 6, 32, device="cuda:0", dtype=torch_dtype)
    actual = model(x=x)["output"]
    assert actual.dtype is torch_dtype
    torch.testing.assert_close(actual.double(), _manual(x, module.weight, module.eps),
                               atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="float16 activations are qualified on CUDA")
def test_the_statistic_is_accumulated_in_float32_so_float16_does_not_overflow():
    plan = resolve("rms_norm(affine=False)", input_shape=("B", 32), output_shape=("B", 32), dtype="float16")
    model, _ = _rms_norm(plan, "cuda:0")
    # x^2 is ~1e5 per element, well past the float16 maximum of 65504.
    x = torch.full((2, 32), 300.0, device="cuda:0", dtype=torch.float16)
    out = model(x=x)["output"]
    assert out.isfinite().all()
    torch.testing.assert_close(out.float(), torch.ones(2, 32, device="cuda:0"), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_non_positive_eps_is_an_argument_error(eps):
    with pytest.raises(HNDLError, match="E_ARGUMENT.*eps"):
        resolve(f"rms_norm({eps})", input_shape=("B", 8), output_shape=("B", 8))


def test_affine_false_has_no_parameters(device):
    plan = resolve("rms_norm(affine=False)", input_shape=("B", 8), output_shape=("B", 8))
    assert parameter_counts(plan) == {plan.nodes[0].id: 0}
    _, module = _rms_norm(plan, device)
    assert module.weight is None
    assert list(module.parameters()) == []


def test_the_default_gain_is_ones_and_construction_settings_override_it(device):
    default = resolve("rms_norm()", input_shape=("B", 8), output_shape=("B", 8))
    assert parameter_counts(default) == {default.nodes[0].id: 8}
    _, module = _rms_norm(default, device)
    torch.testing.assert_close(module.weight, torch.ones(8, device=device))
    zeroed = resolve('rms_norm(init={"weight": 0})', input_shape=("B", 8), output_shape=("B", 8))
    assert zeroed.nodes[0].initialization["overrides"] == {"weight": 0.0}
    model, module = _rms_norm(zeroed, device)
    torch.testing.assert_close(module.weight, torch.zeros(8, device=device))
    out = model(x=torch.randn(2, 8, device=device))["output"]
    torch.testing.assert_close(out, torch.zeros(2, 8, device=device))


def test_width_flows_forward_and_backward_through_the_norm():
    forward = resolve("rms_norm()\nlinear()", input_shape=("B", 16), output_shape=("B", 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 16)
    assert forward.nodes[1].args["in_features"] == 16
    backward = resolve("linear()\nrms_norm()", input_shape=("B", 8), output_shape=("B", 32))
    assert backward.nodes[0].args["out_features"] == 32
    sequence = resolve("linear(8)\nrms_norm()\nlinear(3)", input_shape=("B", 5, 8), output_shape=("B", 5, 3))
    assert sequence.nodes[1].output_shapes["out"] == ("B", 5, 8)


def test_a_contradicting_width_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("rms_norm()", input_shape=("B", 8), output_shape=("B", 12))
