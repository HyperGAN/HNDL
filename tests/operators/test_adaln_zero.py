"""Numerics, inference and error codes for the ``adaln_zero`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build, parameter_counts

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "adaln_zero")


def test_zero_initialization_invariant(device):
    plan = resolve(
        "c = mean(1)\ncond = linear(c, 16)\nnormed, gate = adaln_zero(x, cond)\nlinear(normed, 10)",
        input_shape=("B", 8, 32),
        output_shape=("B", 8, 10),
    )
    model = build(plan, device=device)
    adaln = model[plan.nodes[2].id]
    assert torch.all(adaln.modulation.weight == 0)
    assert torch.all(adaln.modulation.bias == 0)

    x = torch.randn(2, 8, 32, device=device)
    cond = torch.randn(2, 16, device=device)
    out, gate = adaln(x, cond)
    # At initialization, scale=0, shift=0, gate=0
    expected_norm = F.layer_norm(x, (32,), eps=adaln.eps)
    torch.testing.assert_close(out, expected_norm)
    torch.testing.assert_close(gate, torch.zeros_like(x))


def test_modulation_matches_handwritten_formula(device):
    plan = resolve(
        "c = mean(1)\ncond = linear(c, 24)\nnormed, gate = adaln_zero(x, cond)\nmul(normed, gate)",
        input_shape=("B", 5, 16),
        output_shape=("B", 5, 16),
    )
    model = build(plan, device=device)
    adaln = model[plan.nodes[2].id]
    with torch.no_grad():
        adaln.modulation.weight.normal_()
        adaln.modulation.bias.normal_()

    x = torch.randn(3, 5, 16, device=device)
    cond = torch.randn(3, 24, device=device)
    out, gate = adaln(x, cond)

    params = adaln.modulation(F.silu(cond))
    shift, scale, gate_expected = params.chunk(3, dim=-1)
    normed = F.layer_norm(x, (16,), eps=adaln.eps)
    expected_out = (1.0 + scale.unsqueeze(1)) * normed + shift.unsqueeze(1)
    expected_gate = gate_expected.unsqueeze(1).expand_as(x)

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(gate, expected_gate)


def test_parameter_counts():
    plan = resolve(
        "c = mean(1)\ncond = linear(c, 32)\nnormed, gate = adaln_zero(x, cond)\nmul(normed, gate)",
        input_shape=("B", 10, 64),
        output_shape=("B", 10, 64),
    )
    # D_cond=32, D=64 -> 32 * (3 * 64) + (3 * 64) = 6144 + 192 = 6336
    assert parameter_counts(plan)[plan.nodes[2].id] == 6336


@pytest.mark.parametrize("eps", [0.0, -1e-5])
def test_invalid_eps_raises_argument_error(eps):
    with pytest.raises(HNDLError, match="E_ARGUMENT.*eps"):
        resolve(
            f"c = mean(1)\ncond = linear(c, 16)\nnormed, gate = adaln_zero(x, cond, eps={eps})\nmul(normed, gate)",
            input_shape=("B", 8, 32),
            output_shape=("B", 8, 32),
        )


def test_gradcheck_double(device):
    from hndl.operators.adaln_zero import AdaLNZero

    module = AdaLNZero(eps=1e-5, D=8, D_cond=12).double().to(device)
    with torch.no_grad():
        module.modulation.weight.normal_()
        module.modulation.bias.normal_()

    x = torch.randn(2, 4, 8, dtype=torch.float64, device=device, requires_grad=True)
    cond = torch.randn(2, 12, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x, cond), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        "c = mean(1)\ncond = linear(c, 16)\nnormed, gate = adaln_zero(x, cond)\nmul(normed, gate)",
        input_shape=("B", 6, 16),
        output_shape=("B", 6, 16),
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 6, 16, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
