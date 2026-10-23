"""Numerics, inference and error codes for the ``latent_attention`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build, parameter_counts

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "latent_attention")


def test_nope_mode_runs_and_preserves_shape(device):
    plan = resolve(
        "latent_attention(4, 32, qk_rope_head_dim=0, causal=True)",
        input_shape=("B", 10, 64),
        output_shape=("B", 10, 64),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 10, 64, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 10, 64)
    assert out.isfinite().all()


def test_decoupled_rope_mode_runs_and_preserves_shape(device):
    plan = resolve(
        "latent_attention(4, 32, qk_nope_head_dim=16, qk_rope_head_dim=16, causal=True)",
        input_shape=("B", 8, 64),
        output_shape=("B", 8, 64),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 8, 64, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 64)
    assert out.isfinite().all()


def test_q_lora_compression_mode(device):
    plan = resolve(
        "latent_attention(4, 32, q_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=16)",
        input_shape=("B", 6, 64),
        output_shape=("B", 6, 64),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 6, 64, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 6, 64)
    assert out.isfinite().all()


def test_parameter_counts():
    plan = resolve(
        "latent_attention(4, 32, qk_nope_head_dim=16, qk_rope_head_dim=16, bias=False)",
        input_shape=("B", 8, 64),
        output_shape=("B", 8, 64),
    )
    # D=64, heads=4, r_kv=32, d_nope=16, d_rope=16, d_v=16
    # kv_down: 64 * 32 = 2048
    # kv_norm: 32
    # k_up: 32 * (4 * 16) = 2048
    # v_up: 32 * (4 * 16) = 2048
    # k_rope_proj: 64 * 16 = 1024
    # q_up: 64 * (4 * 32) = 8192
    # o_proj: (4 * 16) * 64 = 4096
    # Total: 2048 + 32 + 2048 + 2048 + 1024 + 8192 + 4096 = 19488
    assert parameter_counts(plan) == {"n0": 19488}


def test_odd_qk_rope_head_dim_reports_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*even"):
        resolve(
            "latent_attention(4, 32, qk_rope_head_dim=15)",
            input_shape=("B", 8, 64),
            output_shape=("B", 8, 64),
        )


def test_heads_not_dividing_d_reports_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divide"):
        resolve(
            "latent_attention(5, 32, qk_nope_head_dim=0)",
            input_shape=("B", 8, 64),
            output_shape=("B", 8, 64),
        )


def test_gradcheck_double(device):
    plan = resolve(
        "latent_attention(2, 8, qk_nope_head_dim=8, qk_rope_head_dim=8, causal=True)",
        input_shape=("B", 4, 16),
        output_shape=("B", 4, 16),
    )
    model = build(plan, device=device)
    module = model[plan.nodes[0].id].double()
    x = torch.randn(2, 4, 16, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        "latent_attention(4, 32, qk_nope_head_dim=16, qk_rope_head_dim=16, causal=True)",
        input_shape=("B", 6, 64),
        output_shape=("B", 6, 64),
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 6, 64, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()


def test_weight_absorption_inference_parity(device):
    plan = resolve(
        "latent_attention(4, 32, qk_nope_head_dim=16, qk_rope_head_dim=16, causal=True)",
        input_shape=("B", 8, 64),
        output_shape=("B", 8, 64),
    )
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    x = torch.randn(2, 8, 64, device=device)
    with torch.no_grad():
        out_standard = module(x)
        assert not module.is_weight_absorbed
        module.absorb_weights()
        assert module.is_weight_absorbed
        out_absorbed = module(x)
        torch.testing.assert_close(out_absorbed, out_standard, atol=1e-5, rtol=1e-5)
