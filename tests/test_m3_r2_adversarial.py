"""Adversarial challenger test suite for Milestone 3 Remediation (M3 Gate Iteration 2).

Target operators:
1. joint_attention (mmdit_attention@1):
   - qk_norm="l2_norm" across sequence lengths, head dims, batch sizes, dtypes (fp32, bf16, fp64)
   - Adversarial epsilon handling (zero norm, 1e-12, 1e-25, large values up to 1e20)
   - Numerical parity forward vs _reference (< 1e-5 for fp32/fp64)
   - Backward autograd gradient flow through x, context, weights, biases
   - Execution on CPU and CUDA

2. mamba2 (ssd@1):
   - chunk_size across 16, 32, 64, 128 with varying sequence lengths (T < chunk, T == chunk, T > chunk, non-divisible 73, 117)
   - Numerical parity vs sequential _reference (< 1e-4) across multiple seeds
   - Extreme decay parameters (A near zero, large negative A up to -50.0 and beyond)
   - Joint convolution on [x_ssm, B, C] and causal receptive field verification
   - Backward pass gradient finiteness across all parameters and inputs
"""

import math
import pytest
import torch

from hndl.operators.joint_attention import JointAttention, _reference as joint_attn_ref
from hndl.operators.mamba2 import Mamba2, _reference as mamba2_ref
from hndl.testing import available_devices

DEVICES = available_devices()


# ==============================================================================
# TARGET 1: joint_attention (mmdit_attention@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize("T_x, T_y", [(1, 32), (8, 16), (17, 3), (32, 1), (73, 117)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_joint_attention_l2_norm_sweep(device, batch_size, head_dim, T_x, T_y, dtype):
    """Stress test qk_norm='l2_norm' across sequence lengths, head dimensions, batch sizes, and dtypes."""
    heads = 2
    d_x = heads * head_dim
    d_c = heads * head_dim

    module = JointAttention(
        heads=heads,
        head_dim=head_dim,
        D_x=d_x,
        D_c=d_c,
        qk_norm="l2_norm",
    ).to(device=device, dtype=dtype)
    module.eval()
    ref = joint_attn_ref(module)

    torch.manual_seed(1000 + T_x + T_y + head_dim)
    x = torch.randn(batch_size, T_x, d_x, device=device, dtype=dtype, requires_grad=True)
    ctx = torch.randn(batch_size, T_y, d_c, device=device, dtype=dtype, requires_grad=True)

    out, ctx_out = module(x, ctx)
    ref_out, ref_ctx = ref(x, ctx)

    # Output shape verification
    assert out.shape == (batch_size, T_x, d_x)
    assert ctx_out.shape == (batch_size, T_y, d_c)

    # Numerical parity verification
    diff_x = (out - ref_out).abs().max().item()
    diff_c = (ctx_out - ref_ctx).abs().max().item()

    # fp64/fp32 bound is strictly < 1e-5; bf16 has 7-bit mantissa (epsilon ~ 7.8e-3), bound < 2.5e-3
    tol = 1e-5 if dtype in (torch.float32, torch.float64) else 2.5e-3
    assert diff_x < tol, f"diff_x={diff_x} exceeds tol={tol} on {device} {dtype} head_dim={head_dim}"
    assert diff_c < tol, f"diff_c={diff_c} exceeds tol={tol} on {device} {dtype} head_dim={head_dim}"

    # Gradient flow verification
    loss = out.sum() + ctx_out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), f"x.grad has non-finite values on {device}"
    assert ctx.grad is not None and torch.isfinite(ctx.grad).all(), f"ctx.grad has non-finite values on {device}"
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"param {name}.grad non-finite on {device}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [0.0, 1e-12, 1e-25, 1e6, 1e12, 1e20])
def test_joint_attention_adversarial_epsilon_and_extremes(device, scale):
    """Adversarially verify epsilon handling when queries/keys have zero, near-zero, or large norms."""
    module = JointAttention(heads=2, head_dim=16, D_x=32, D_c=32, qk_norm="l2_norm").to(device)

    torch.manual_seed(2026)
    x = (torch.randn(2, 8, 32, device=device) * scale).requires_grad_(True)
    ctx = (torch.randn(2, 12, 32, device=device) * scale).requires_grad_(True)

    out, ctx_out = module(x, ctx)
    assert torch.isfinite(out).all(), f"out non-finite with input scale {scale} on {device}"
    assert torch.isfinite(ctx_out).all(), f"ctx_out non-finite with input scale {scale} on {device}"

    loss = out.sum() + ctx_out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), f"x.grad non-finite with scale {scale} on {device}"
    assert ctx.grad is not None and torch.isfinite(ctx.grad).all(), f"ctx.grad non-finite with scale {scale} on {device}"
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"param {name}.grad non-finite with scale {scale}"


@pytest.mark.parametrize("device", DEVICES)
def test_joint_attention_zero_qk_projections(device):
    """Verify that zero-projected queries and keys produce 100% finite outputs and gradients."""
    module = JointAttention(heads=2, head_dim=16, D_x=32, D_c=32, qk_norm="l2_norm", bias=False).to(device)
    with torch.no_grad():
        module.x_q_proj.weight.zero_()
        module.x_k_proj.weight.zero_()
        module.c_q_proj.weight.zero_()
        module.c_k_proj.weight.zero_()

    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    ctx = torch.randn(2, 12, 32, device=device, requires_grad=True)

    out, ctx_out = module(x, ctx)
    assert torch.isfinite(out).all()
    assert torch.isfinite(ctx_out).all()

    loss = out.sum() + ctx_out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(ctx.grad).all()
    for name, p in module.named_parameters():
        assert torch.isfinite(p.grad).all(), f"param {name}.grad non-finite on zero QK"


@pytest.mark.parametrize("device", DEVICES)
def test_joint_attention_edge_empty_shapes(device):
    """Verify empty batch B=0 and empty sequence T_x=0 or T_c=0 handling."""
    module = JointAttention(heads=2, head_dim=16, D_x=32, D_c=32, qk_norm="l2_norm").to(device)

    # B=0
    x_b0 = torch.randn(0, 8, 32, device=device)
    c_b0 = torch.randn(0, 16, 32, device=device)
    out_b0, ctx_b0 = module(x_b0, c_b0)
    assert out_b0.shape == (0, 8, 32)
    assert ctx_b0.shape == (0, 16, 32)

    # T_x=0
    x_t0 = torch.randn(2, 0, 32, device=device)
    c_t1 = torch.randn(2, 16, 32, device=device)
    out_t0, ctx_t1 = module(x_t0, c_t1)
    assert out_t0.shape == (2, 0, 32)
    assert ctx_t1.shape == (2, 16, 32)

    # T_c=0
    x_t2 = torch.randn(2, 16, 32, device=device)
    c_t0 = torch.randn(2, 0, 32, device=device)
    out_t2, ctx_t0 = module(x_t2, c_t0)
    assert out_t2.shape == (2, 16, 32)
    assert ctx_t0.shape == (2, 0, 32)


# ==============================================================================
# TARGET 2: mamba2 (ssd@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
@pytest.mark.parametrize("seed", [42, 1337, 2026])
def test_mamba2_chunk_size_and_sequence_lengths(device, chunk_size, seed):
    """Stress test chunk_size across sequence lengths and verify reference parity strictly < 1e-4."""
    torch.manual_seed(seed)
    seq_lens = [
        chunk_size // 2,
        chunk_size,
        chunk_size + 7,
        2 * chunk_size,
        2 * chunk_size + 11,
        73,
        117,
    ]

    for T in seq_lens:
        module = Mamba2(d_model=32, heads=4, head_dim=16, chunk_size=chunk_size).to(device)
        module.eval()
        ref = mamba2_ref(module)

        x = torch.randn(2, T, 32, device=device)
        out = module(x)
        ref_out = ref(x)

        diff = (out - ref_out).abs().max().item()
        assert diff < 1e-4, f"Parity diff {diff} exceeded 1e-4 at seed={seed}, cs={chunk_size}, T={T} on {device}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name, a_log_val",
    [
        ("A near zero (A_log = -50)", -50.0),
        ("A = -1.0 (A_log = 0)", 0.0),
        ("A = -10.0 (A_log = ln(10))", math.log(10.0)),
        ("A = -50.0 (A_log = ln(50))", math.log(50.0)),
        ("A = -100.0 (A_log = ln(100))", math.log(100.0)),
        ("Extreme negative A (A_log = 10.0, a_scaled ~ -22000)", 10.0),
    ],
)
def test_mamba2_extreme_decay_parameters(device, name, a_log_val):
    """Test extreme decay parameters and verify no NaNs in forward or backward autograd."""
    module = Mamba2(d_model=32, heads=4, head_dim=16, chunk_size=32).to(device)
    with torch.no_grad():
        module.A_log.fill_(a_log_val)

    x = torch.randn(2, 73, 32, device=device, requires_grad=True)
    out = module(x)
    assert torch.isfinite(out).all(), f"{name} on {device} produced non-finite forward output"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), f"{name} on {device} produced non-finite x.grad"
    for pname, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name} on {device} produced non-finite {pname}.grad"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d_conv", [2, 4, 7])
def test_mamba2_joint_convolution_and_causal_receptive_field(device, d_conv):
    """Stress test joint conv on [x_ssm, B, C] and verify receptive field is causal."""
    module = Mamba2(d_model=32, heads=4, head_dim=16, d_conv=d_conv, chunk_size=32).to(device)
    module.eval()

    T = 60
    t_split = 25

    # 1. Forward causality verification: changing future inputs t >= t_split must not alter outputs t < t_split
    x1 = torch.randn(2, T, 32, device=device)
    x2 = x1.clone()
    x2[:, t_split:] = torch.randn(2, T - t_split, 32, device=device)

    out1 = module(x1)
    out2 = module(x2)

    diff_past = (out1[:, :t_split] - out2[:, :t_split]).abs().max().item()
    assert diff_past == 0.0, f"Forward causality violated on {device} with d_conv={d_conv}: diff={diff_past}"

    # 2. Backward causality verification: loss at t_loss must not induce gradients on future inputs t > t_loss
    t_loss = 20
    x = torch.randn(2, T, 32, device=device, requires_grad=True)
    out = module(x)
    loss = out[:, t_loss].sum()
    loss.backward()

    assert x.grad is not None
    future_grad = x.grad[:, t_loss + 1 :].abs().max().item()
    assert future_grad == 0.0, f"Backward causality violated on {device} with d_conv={d_conv}: future_grad={future_grad}"

    # 3. Joint convolution verification: verify conv1d weights affect x_ssm, B, and C channels
    inner_dim = module.inner_dim
    bc_dim = module.heads * module.d_state
    conv_w_grad = module.conv1d.weight.grad
    assert conv_w_grad[:inner_dim].abs().sum().item() > 0, "conv1d did not affect x_ssm channels"
    assert conv_w_grad[inner_dim : inner_dim + bc_dim].abs().sum().item() > 0, "conv1d did not affect B channels"
    assert conv_w_grad[inner_dim + bc_dim :].abs().sum().item() > 0, "conv1d did not affect C channels"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("conv_bias", [True, False])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("T", [1, 16, 33, 73])
def test_mamba2_backward_gradient_finiteness_exhaustive(device, bias, conv_bias, chunk_size, T):
    """Verify backward pass gradients are strictly finite across all parameters and inputs."""
    module = Mamba2(
        d_model=32,
        heads=4,
        head_dim=16,
        chunk_size=chunk_size,
        bias=bias,
        conv_bias=conv_bias,
    ).to(device)

    x = torch.randn(2, T, 32, device=device, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None, f"Parameter {name}.grad is None on {device}"
        assert torch.isfinite(p.grad).all(), f"Parameter {name}.grad is non-finite on {device}"
