"""Numerics, inference and error codes for the ``grn`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build, parameter_counts

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "grn")


def test_at_initialization_grn_is_an_exact_identity(device):
    plan = resolve("grn()", input_shape=("B", 16, 8, 8), output_shape=("B", 16, 8, 8))
    model = build(plan, device=device)
    x = torch.randn(2, 16, 8, 8, device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, x)


def test_matches_mathematical_formula_with_learned_parameters(device):
    plan = resolve("grn(eps=1e-5)", input_shape=("B", 8, 4, 4), output_shape=("B", 8, 4, 4))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    with torch.no_grad():
        module.gamma.normal_()
        module.beta.normal_()
    x = torch.randn(3, 8, 4, 4, device=device)
    gx = torch.sqrt(torch.sum(x**2, dim=(2, 3), keepdim=True) + 1e-5)
    nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-5)
    expected = module.gamma * (x * nx) + module.beta + x
    out = model(x=x)["output"]
    torch.testing.assert_close(out, expected)


def test_parameter_counts():
    plan = resolve("grn()", input_shape=("B", 32, 7, 7), output_shape=("B", 32, 7, 7))
    assert parameter_counts(plan) == {"n0": 64}


def test_channels_and_spatial_dimensions_flow_forward_and_backward():
    forward = resolve("conv(16, kernel_size=1)\ngrn()", input_shape=("B", 8, 4, 4), output_shape=("B", 16, 4, 4))
    assert forward.nodes[1].output_shapes["out"] == ("B", 16, 4, 4)
    backward = resolve("grn()\nconv(10, kernel_size=1)", input_shape=("B", 24, 6, 6), output_shape=("B", 10, 6, 6))
    assert backward.nodes[0].output_shapes["out"] == ("B", 24, 6, 6)


def test_rejects_non_rank_4_tensors():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("grn()", input_shape=("B", 16), output_shape=("B", 16))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("grn()", input_shape=("B", 10, 16), output_shape=("B", 10, 16))


@pytest.mark.parametrize("eps", [0.0, -1e-5])
def test_invalid_eps_raises_argument_error(eps):
    with pytest.raises(HNDLError, match="E_ARGUMENT.*eps"):
        resolve(f"grn(eps={eps})", input_shape=("B", 8, 4, 4), output_shape=("B", 8, 4, 4))


def test_gradcheck_double(device):
    plan = resolve("grn(eps=1e-5)", input_shape=("B", 4, 3, 3), output_shape=("B", 4, 3, 3))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    with torch.no_grad():
        module.gamma.normal_()
        module.beta.normal_()
    x = torch.randn(2, 4, 3, 3, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve("grn()", input_shape=("B", 8, 4, 4), output_shape=("B", 8, 4, 4), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 8, 4, 4, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
