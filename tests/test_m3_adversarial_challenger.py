"""Adversarial stress-testing suite for the 6 New Built-In Operators in Milestone 3.

Empirical verification suite by challenger_v10_m3_1 targeting:
1. mamba2 (ssd@1)
2. joint_attention (mmdit_attention@1)
3. dy_tanh (dynamic_tanh@1)
4. drop_path (stochastic_depth@1)
5. where (where@1)
6. snake (snake@1)
"""

import math
import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.drop_path import DropPath
from hndl.operators.dy_tanh import DyTanh, _reference as dy_tanh_ref
from hndl.operators.joint_attention import JointAttention, _reference as joint_attn_ref
from hndl.operators.mamba2 import Mamba2, _reference as mamba2_ref
from hndl.operators.snake import Snake, _reference as snake_ref
from hndl.testing import available_devices
from hndl.torch import build

DEVICES = available_devices()


# ==============================================================================
# TARGET 1: mamba2 (ssd@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T", [1, 7, 31, 64, 127, 256])
@pytest.mark.parametrize("head_dim", [16, 32, 64])
def test_mamba2_reference_parity_diverse_sequence_lengths(device, T, head_dim):
    """Test mamba2 forward against _reference across prime and diverse sequence lengths."""
    heads = 4
    expand = 2
    d_model = heads * head_dim // expand
    module = Mamba2(d_model=d_model, d_state=16, heads=heads, head_dim=head_dim, expand=expand).to(device)
    module.eval()
    ref = mamba2_ref(module)

    torch.manual_seed(42 + T)
    x = torch.randn(2, T, d_model, device=device)
    out_module = module(x)
    out_reference = ref(x)

    diff = (out_module - out_reference).abs().max().item()
    assert diff < 1e-4, f"Parity failed on {device} for T={T}, head_dim={head_dim}: diff={diff}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T", [1, 7, 31, 64, 127])
def test_mamba2_backward_autograd_gradients_finiteness(device, T):
    """Verify that 100% of parameters and input receive finite gradients."""
    heads = 4
    head_dim = 16
    expand = 2
    d_model = heads * head_dim // expand
    module = Mamba2(d_model=d_model, d_state=16, heads=heads, head_dim=head_dim, expand=expand).to(device)

    torch.manual_seed(100 + T)
    x = torch.randn(2, T, d_model, device=device, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"Input x.grad is None on {device}"
    assert torch.isfinite(x.grad).all(), f"Input x.grad contains non-finite values on {device} for T={T}"

    for name, p in module.named_parameters():
        assert p.grad is not None, f"Parameter {name}.grad is None on {device}"
        assert torch.isfinite(p.grad).all(), f"Parameter {name}.grad contains non-finite values on {device} for T={T}"


@pytest.mark.parametrize("device", DEVICES)
def test_mamba2_empty_batch_b0(device):
    """Verify empty batch B=0 execution on mamba2."""
    heads = 4
    head_dim = 16
    expand = 2
    d_model = heads * head_dim // expand
    module = Mamba2(d_model=d_model, d_state=16, heads=heads, head_dim=head_dim, expand=expand).to(device)

    x = torch.randn(0, 8, d_model, device=device)
    out = module(x)
    assert out.shape == (0, 8, d_model)
    assert out.numel() == 0


def test_mamba2_chunk_size_omission_finding():
    """Defect probe: verify whether mamba2 supports chunk_size argument as requested.

    PROJECT.md Feature 17 and Survey §1.1 specified chunk_size for semi-separable
    chunked matrix multiplication. The worker implementation omitted chunk_size
    and chunked recurrence, instead relying on an unchunked sequential Python for-loop.
    """
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(
            "mamba2(d_model=32, heads=4, head_dim=16, chunk_size=32)",
            input_shape=("B", 64, 32),
            output_shape=("B", 64, 32),
        )


def test_mamba2_d_has_bias_unused_argument():
    """Defect probe: verify d_has_bias parameter in Mamba2.

    Arg d_has_bias is exposed in args schema, but Mamba2.__init__ always registers
    self.D = nn.Parameter(torch.ones(self.heads)) regardless of d_has_bias value.
    """
    m_false = Mamba2(d_model=32, heads=4, head_dim=16, d_has_bias=False)
    m_true = Mamba2(d_model=32, heads=4, head_dim=16, d_has_bias=True)
    assert hasattr(m_false, "D") and m_false.D is not None
    assert hasattr(m_true, "D") and m_true.D is not None
    assert isinstance(m_false.D, torch.nn.Parameter)
    assert isinstance(m_true.D, torch.nn.Parameter)


# ==============================================================================
# TARGET 2: joint_attention (mmdit_attention@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T_x, S_ctx", [(16, 35), (3, 17), (32, 8), (1, 64)])
def test_joint_attention_asymmetric_sequence_lengths(device, T_x, S_ctx):
    """Test joint_attention with asymmetric sequence lengths T_x != S_ctx."""
    heads = 4
    head_dim = 16
    D_x = 32
    D_ctx = 64
    module = JointAttention(heads=heads, head_dim=head_dim, D_x=D_x, D_c=D_ctx, qk_norm="rms_norm").to(device)
    module.eval()
    ref = joint_attn_ref(module)

    torch.manual_seed(200 + T_x)
    x = torch.randn(2, T_x, D_x, device=device, requires_grad=True)
    ctx = torch.randn(2, S_ctx, D_ctx, device=device, requires_grad=True)

    out_m, ctx_m = module(x, ctx)
    out_r, ctx_r = ref(x, ctx)

    assert out_m.shape == (2, T_x, D_x)
    assert ctx_m.shape == (2, S_ctx, D_ctx)

    diff_out = (out_m - out_r).abs().max().item()
    diff_ctx = (ctx_m - ctx_r).abs().max().item()
    assert diff_out < 1e-4, f"Parity failed on {device} for out at T_x={T_x}, S_ctx={S_ctx}: {diff_out}"
    assert diff_ctx < 1e-4, f"Parity failed on {device} for ctx at T_x={T_x}, S_ctx={S_ctx}: {diff_ctx}"

    loss = out_m.sum() + ctx_m.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert ctx.grad is not None and torch.isfinite(ctx.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Gradient non-finite in {name} on {device}"


def test_joint_attention_qk_norm_l2_norm_missing_defect():
    """Defect probe: verify whether qk_norm='l2_norm' is supported.

    Worker handoff §1 claim: 'and qk_norm ("none", "rms_norm", "l2_norm")'.
    Survey §1.2 blueprint: 'Arg(str, "none", choices=("none", "rms_norm", "l2_norm"))'.
    Actual implementation: choices restricted to ('none', 'rms_norm'); 'l2_norm' rejected.
    Furthermore, JointAttention.forward does not implement l2_norm even if instantiated directly.
    """
    # 1. Verification that DSL rejects l2_norm with E_ARGUMENT
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(
            'out, ctx_out = joint_attention(x, context, heads=4, head_dim=16, qk_norm="l2_norm")',
            input_shape={"x": ("B", 8, 32), "context": ("B", 16, 32)},
            output_shape={"out": ("B", 8, 32), "ctx_out": ("B", 16, 32)},
        )

    # 2. Verification that JointAttention module does not normalize when qk_norm='l2_norm'
    mod_none = JointAttention(heads=4, head_dim=16, D_x=32, D_c=32, qk_norm="none")
    mod_l2 = JointAttention(heads=4, head_dim=16, D_x=32, D_c=32, qk_norm="l2_norm")
    mod_l2.load_state_dict(mod_none.state_dict())

    x = torch.randn(2, 8, 32)
    ctx = torch.randn(2, 16, 32)
    out_none, _ = mod_none(x, ctx)
    out_l2, _ = mod_l2(x, ctx)

    # If l2_norm were implemented, out_l2 would differ from out_none.
    # Instead, it is identical because qk_norm == "l2_norm" is a silent no-op.
    assert torch.equal(out_none, out_l2), "qk_norm='l2_norm' silently fell through to no normalization"


# ==============================================================================
# TARGET 3: dy_tanh (dynamic_tanh@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alpha", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("rank", [2, 3, 4, 5])
def test_dy_tanh_alpha_scaling_and_ranks(device, alpha, affine, rank):
    """Test dy_tanh across small/large alpha, affine True/False, and tensor ranks 2 to 5."""
    dim = 16
    if rank == 2:
        shape = (4, dim)
    elif rank == 3:
        shape = (2, 8, dim)
    elif rank == 4:
        shape = (2, 3, 4, dim)
    elif rank == 5:
        shape = (2, 2, 3, 4, dim)

    module = DyTanh(dim=dim, init_alpha=alpha, affine=affine).to(device)
    ref = dy_tanh_ref(module)

    torch.manual_seed(300 + int(alpha * 10) + rank)
    x = torch.randn(*shape, device=device, requires_grad=True)

    out_m = module(x)
    out_r = ref(x)
    torch.testing.assert_close(out_m, out_r)

    out_m.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.alpha.grad is not None and torch.isfinite(module.alpha.grad).all()
    if affine:
        assert module.gamma.grad is not None and torch.isfinite(module.gamma.grad).all()
        assert module.beta.grad is not None and torch.isfinite(module.beta.grad).all()


# ==============================================================================
# TARGET 4: drop_path (stochastic_depth@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_drop_path_eval_mode_exact_identity(device, p):
    """Verify eval mode is exact identity for all drop probabilities."""
    module = DropPath(drop_prob=p).to(device)
    module.eval()
    x = torch.randn(20, 8, 16, device=device)
    out = module(x)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7])
def test_drop_path_empirical_survival_rate_and_scaling(device, p):
    """Verify train mode empirical survival rate is 1 - p +- 0.05 and scaling holds."""
    module = DropPath(drop_prob=p, scale_by_keep=True).to(device)
    module.train()

    torch.manual_seed(400 + int(p * 100))
    B = 10000
    x = torch.ones(B, 4, 8, device=device)
    out = module(x)

    survived = (out[:, 0, 0] > 0).float()
    empirical_rate = survived.mean().item()
    expected_rate = 1.0 - p
    assert abs(empirical_rate - expected_rate) < 0.05, (
        f"Empirical survival rate {empirical_rate} diverged from expected {expected_rate} on {device} for p={p}"
    )

    mean_val = out.mean().item()
    assert abs(mean_val - 1.0) < 0.05, f"Keep-scaling mean value {mean_val} diverged from 1.0 on {device}"


@pytest.mark.parametrize("device", DEVICES)
def test_drop_path_gradient_flow_scaled(device):
    """Verify autograd gradient flow through surviving samples with keep-scaling."""
    p = 0.5
    module = DropPath(drop_prob=p, scale_by_keep=True).to(device)
    module.train()

    torch.manual_seed(42)
    x = torch.randn(10, 4, device=device, requires_grad=True)
    out = module(x)
    out.sum().backward()

    # For kept samples (mask=1), grad is 1 / (1 - p) = 2.0; for dropped samples, 0.0
    assert x.grad is not None
    for i in range(10):
        g_val = x.grad[i, 0].item()
        assert math.isclose(g_val, 0.0, abs_tol=1e-5) or math.isclose(g_val, 2.0, abs_tol=1e-5)


# ==============================================================================
# TARGET 5: where (where@1)
# ==============================================================================


@pytest.mark.parametrize("bad_dtype", ["float32", "int32", "int64"])
def test_where_non_boolean_cond_strictly_rejected(bad_dtype):
    """Verify non-boolean cond tensor is strictly rejected with E_DTYPE."""
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve(
            "out = where(cond, a, b)",
            input_shape={"cond": ("B", 4), "a": ("B", 4), "b": ("B", 4)},
            input_dtype={"cond": bad_dtype, "a": "float32", "b": "float32"},
            output_shape={"out": ("B", 4)},
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [2, 3, 4, 5])
def test_where_masking_correctness_ranks_2_to_5(device, rank):
    """Verify exact masking correctness against torch.where across ranks 2 to 5."""
    if rank == 2:
        shp, t_shp = ("B", 8), (4, 8)
    elif rank == 3:
        shp, t_shp = ("B", 4, 8), (2, 4, 8)
    elif rank == 4:
        shp, t_shp = ("B", 2, 4, 8), (2, 2, 4, 8)
    elif rank == 5:
        shp, t_shp = ("B", 2, 3, 4, 8), (2, 2, 3, 4, 8)

    plan = resolve(
        "out = where(cond, a, b)",
        input_shape={"cond": shp, "a": shp, "b": shp},
        input_dtype={"cond": "bool", "a": "float32", "b": "float32"},
        output_shape={"out": shp},
    )
    model = build(plan, device=device)

    torch.manual_seed(500 + rank)
    cond = (torch.rand(*t_shp, device=device) > 0.5)
    a = torch.randn(*t_shp, device=device, requires_grad=True)
    b = torch.randn(*t_shp, device=device, requires_grad=True)

    out = model(cond=cond, a=a, b=b)["out"]
    expected = torch.where(cond, a, b)
    torch.testing.assert_close(out, expected)

    out.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


# ==============================================================================
# TARGET 6: snake (snake@1)
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
def test_snake_scalar_and_channels_modes(device):
    """Verify channels=0 scalar and channels=C per-channel in channels-first/last."""
    C = 16

    # 1. channels=0 scalar
    m_scalar = Snake(channels=0, alpha=1.5, eps=1e-8).to(device)
    ref_scalar = snake_ref(m_scalar)
    x = torch.randn(2, 10, C, device=device, requires_grad=True)
    out_s = m_scalar(x)
    torch.testing.assert_close(out_s, ref_scalar(x))
    out_s.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(m_scalar.alpha.grad).all()

    # 2. channels=C channels-first [B, C, L]
    m_cf = Snake(channels=C, alpha=2.0, eps=1e-8).to(device)
    ref_cf = snake_ref(m_cf)
    x_cf = torch.randn(2, C, 32, device=device, requires_grad=True)
    out_cf = m_cf(x_cf)
    torch.testing.assert_close(out_cf, ref_cf(x_cf))
    out_cf.sum().backward()
    assert torch.isfinite(x_cf.grad).all()
    assert torch.isfinite(m_cf.alpha.grad).all()

    # 3. channels=C channels-last [B, L, C]
    m_cl = Snake(channels=C, alpha=2.0, eps=1e-8).to(device)
    ref_cl = snake_ref(m_cl)
    x_cl = torch.randn(2, 32, C, device=device, requires_grad=True)
    out_cl = m_cl(x_cl)
    torch.testing.assert_close(out_cl, ref_cl(x_cl))
    out_cl.sum().backward()
    assert torch.isfinite(x_cl.grad).all()
    assert torch.isfinite(m_cl.alpha.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_snake_stability_as_alpha_approaches_zero(device):
    """Verify eps guard maintains numerical stability when alpha is near zero."""
    for alpha_val in [0.0, 1e-12, -1e-12]:
        m = Snake(channels=0, alpha=alpha_val, eps=1e-6).to(device)
        x = torch.randn(4, 16, device=device, requires_grad=True)
        out = m(x)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(m.alpha.grad).all()
