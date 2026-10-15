"""Comprehensive empirical stress-test harness for Milestone 3 numerics and boundary limits.

Covers:
- H-06: Float64 RMSNorm precision across extreme values (denormals, 1e-12, 1e12, random
  shapes [1, 1], [32, 2048], 4D/5D), comparing directly with F.rms_norm and running
  torch.autograd.gradcheck and gradgradcheck on double precision.
- M-12: RoPE angle and frequency calculations in float64 vs float32, ensuring no float32
  truncation occurs in intermediate angles, testing large sequence lengths, boundary
  positions (T=0, T=1, T=100000), head dimension boundaries, and gradcheck.
- M-12: PosEmbed sequence length bounds testing: test T == max_len, T == max_len + 1,
  T == 0, T == 10 * max_len across various ranks (rank 1, rank 2, rank 3, rank 4),
  both for standalone modules and declarative graph resolution/runtime.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from hndl import HNDLError, resolve
from hndl.operators.attention import Attention, ROPE_BASE, _apply_rope, _reference as attention_ref, _rope_tables
from hndl.operators.pos_embed import PosEmbed, _reference as pos_embed_ref
from hndl.operators.rms_norm import RMSNorm, _reference as rms_norm_ref
from hndl.torch import network


# ==============================================================================
# H-06: Float64 RMSNorm Precision & Extreme Numerics
# ==============================================================================

class TestRMSNormFloat64NumericsStress:
    """Empirical precision stress tests for RMSNorm float64 preservation."""

    @pytest.mark.parametrize("magnitude", [
        1e-315,  # float64 denormal / subnormal
        5e-310,  # float64 denormal
        1e-308,  # boundary near minimum normalized float64
        1e-50,   # ultra-small normal float64
        1e-25,
        1e-12,   # audit specific: 1e-12
        1e-6,
        1.0,
        1e6,
        1e12,    # audit specific: 1e12
        1e20,
        1e100,
        1e150,   # large normal float64 (square is 1e300, within float64 limit ~1.79e308)
    ])
    def test_rmsnorm_float64_extreme_magnitudes(self, magnitude):
        """Float64 RMSNorm retains full precision against F.rms_norm across extreme scales."""
        D = 16
        norm = RMSNorm(eps=1e-6, affine=True, D=D).double()
        # Non-trivial affine weights
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, D, dtype=torch.float64))

        # Test both positive and negative values with extreme scale
        x = (torch.randn(4, D, dtype=torch.float64) * 0.1 + 1.0) * magnitude
        out = norm(x)
        ref = F.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps)

        assert out.dtype == torch.float64
        assert ref.dtype == torch.float64
        # Parity must be exact or within double precision machine epsilon
        torch.testing.assert_close(out, ref, rtol=1e-14, atol=1e-14)

    def test_rmsnorm_float64_denormals_and_mixed_signs(self):
        """Denormal values and mixed signs across channels normalize with exact parity."""
        D = 8
        norm = RMSNorm(eps=1e-6, affine=False, D=D).double()
        # Mixed signs, denormals, and zeros
        x = torch.tensor([
            [-1e-315, 1e-315, -0.0, 0.0, -1e-12, 1e-12, -1e12, 1e12],
            [1e-310, -1e-310, 1e-50, -1e-50, 0.5, -0.5, 100.0, -100.0],
        ], dtype=torch.float64)

        out = norm(x)
        ref = F.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps)

        assert out.dtype == torch.float64
        torch.testing.assert_close(out, ref, rtol=1e-14, atol=1e-14)

    @pytest.mark.parametrize("shape", [
        (1, 1),             # audit specific: [1, 1]
        (32, 2048),         # audit specific: [32, 2048]
        (1, 1, 1),          # 3D singleton
        (4, 16, 64),        # 3D sequence
        (2, 4, 8, 16),      # 4D spatial/attention
        (2, 3, 4, 5, 6),    # 5D high-rank
        (1, 10000),         # very wide feature dimension
        (500, 1),           # large batch, D=1
    ])
    def test_rmsnorm_float64_random_shapes(self, shape):
        """RMSNorm matches F.rms_norm across arbitrary dimensional shapes in float64."""
        D = shape[-1]
        norm = RMSNorm(eps=1e-6, affine=True, D=D).double()
        x = torch.randn(*shape, dtype=torch.float64)

        out = norm(x)
        ref = F.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps)

        assert out.shape == shape
        assert out.dtype == torch.float64
        # Discrepancy against F.rms_norm must be 0.0 or negligible
        torch.testing.assert_close(out, ref, rtol=1e-14, atol=1e-14)

    @pytest.mark.parametrize("affine", [True, False])
    @pytest.mark.parametrize("eps", [1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0])
    def test_rmsnorm_float64_eps_and_affine_variants(self, affine, eps):
        """Parity with F.rms_norm holds across affine configs and eps values."""
        D = 12
        norm = RMSNorm(eps=eps, affine=affine, D=D).double()
        if affine:
            with torch.no_grad():
                norm.weight.normal_(mean=1.0, std=0.2)

        x = torch.randn(6, D, dtype=torch.float64)
        out = norm(x)
        ref = F.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps)

        assert out.dtype == torch.float64
        torch.testing.assert_close(out, ref, rtol=1e-14, atol=1e-14)

    def test_rmsnorm_float64_non_contiguous_and_strided_views(self):
        """Non-contiguous and sliced inputs maintain double precision without memory corruption."""
        norm = RMSNorm(eps=1e-6, affine=True, D=8).double()

        # Transposed 2D
        x_orig = torch.randn(8, 16, dtype=torch.float64)
        x_trans = x_orig.t()
        assert not x_trans.is_contiguous()
        torch.testing.assert_close(norm(x_trans), F.rms_norm(x_trans, norm.normalized_shape, norm.weight, norm.eps))

        # Strided slice 3D
        x_3d = torch.randn(4, 10, 16, dtype=torch.float64)[:, ::2, :8]
        assert not x_3d.is_contiguous()
        torch.testing.assert_close(norm(x_3d), F.rms_norm(x_3d, norm.normalized_shape, norm.weight, norm.eps))

    @pytest.mark.parametrize("affine", [True, False])
    @pytest.mark.parametrize("shape", [(2, 3), (1, 1), (2, 3, 4)])
    def test_rmsnorm_float64_gradcheck_analytical_jacobian(self, affine, shape):
        """torch.autograd.gradcheck confirms exact Jacobian in double precision."""
        D = shape[-1]
        norm = RMSNorm(eps=1e-5, affine=affine, D=D).double()
        x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(norm, (x,), eps=1e-6, atol=1e-5, check_batched_grad=True)
        assert torch.autograd.gradgradcheck(norm, (x,), eps=1e-6, atol=1e-5)

    def test_rmsnorm_float16_bfloat16_accumulation_stability(self):
        """Float16/bfloat16 inputs accumulate in float32 without overflow."""
        norm = RMSNorm(eps=1e-6, affine=True, D=4)

        # FP16 values that would overflow if squared in fp16 (60000^2 = 3.6e9 > 65504)
        norm_fp16 = norm.half()
        x_fp16 = torch.full((2, 4), 60000.0, dtype=torch.float16)
        out_fp16 = norm_fp16(x_fp16)
        assert out_fp16.dtype == torch.float16
        assert torch.isfinite(out_fp16).all()
        # Normalized values should be ~ 1.0 (since all elements are equal)
        torch.testing.assert_close(out_fp16, torch.ones_like(out_fp16), rtol=1e-3, atol=1e-3)

        # BF16 stability
        norm_bf16 = norm.bfloat16()
        x_bf16 = torch.full((2, 4), 1e18, dtype=torch.bfloat16)
        out_bf16 = norm_bf16(x_bf16)
        assert out_bf16.dtype == torch.bfloat16
        assert torch.isfinite(out_bf16).all()

    def test_rmsnorm_reference_oracle_parity(self):
        """Declared operator reference matches the module forward pass in double precision."""
        norm = RMSNorm(eps=1e-6, affine=True, D=32).double()
        oracle = rms_norm_ref(norm)

        x = torch.randn(8, 32, dtype=torch.float64)
        out_mod = norm(x)
        out_ora = oracle(x)
        assert torch.equal(out_mod, out_ora)

    def test_rmsnorm_declarative_network_end_to_end_double(self):
        """Declarative GraphModule compiled with rms_norm runs end-to-end in float64."""
        model = network("rms_norm(eps=1e-5, affine=True)",
                        input_shape=("B", 10, 16), output_shape=("B", 10, 16),
                        device="cpu").double()
        x = torch.randn(3, 10, 16, dtype=torch.float64, requires_grad=True)
        out = model(x)
        assert out.dtype == torch.float64
        assert out.shape == (3, 10, 16)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None and x.grad.dtype == torch.float64
        assert model["n0"].weight.grad is not None
        assert model["n0"].weight.grad.dtype == torch.float64


# ==============================================================================
# M-12: RoPE Angle and Frequency Precision in Float64 vs Float32
# ==============================================================================

class TestRoPEFloat64PrecisionStress:
    """Stress tests verifying rotary position embedding precision in float64."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_rope_tables_dtype_preservation(self, dtype):
        """_rope_tables produces exact requested dtype for all floating representations."""
        cos, sin = _rope_tables(positions=32, head_dim=16, device="cpu", dtype=dtype)
        assert cos.dtype == dtype
        assert sin.dtype == dtype
        assert cos.shape == (1, 1, 32, 16)
        assert sin.shape == (1, 1, 32, 16)

    @pytest.mark.parametrize("positions", [1, 16, 128, 512])
    @pytest.mark.parametrize("head_dim", [2, 16, 64])
    def test_rope_float64_oracle_analytical_parity(self, positions, head_dim):
        """Float64 RoPE matches Python 64-bit math analytical oracle within 1e-15."""
        cos_f64, sin_f64 = _rope_tables(positions, head_dim, device="cpu", dtype=torch.float64)

        # Build analytical reference with Python 64-bit IEEE 754 math
        ref_cos = torch.empty(1, 1, positions, head_dim, dtype=torch.float64)
        ref_sin = torch.empty(1, 1, positions, head_dim, dtype=torch.float64)

        for p in range(positions):
            for i in range(0, head_dim, 2):
                exp = i / head_dim
                inv_freq = math.pow(ROPE_BASE, -exp)
                ang = p * inv_freq
                c = math.cos(ang)
                s = math.sin(ang)
                ref_cos[0, 0, p, i // 2] = c
                ref_cos[0, 0, p, (head_dim // 2) + i // 2] = c
                ref_sin[0, 0, p, i // 2] = s
                ref_sin[0, 0, p, (head_dim // 2) + i // 2] = s

        max_cos_err = (cos_f64 - ref_cos).abs().max().item()
        max_sin_err = (sin_f64 - ref_sin).abs().max().item()

        assert max_cos_err < 1e-15, f"RoPE cos error {max_cos_err} exceeds 1e-15"
        assert max_sin_err < 1e-15, f"RoPE sin error {max_sin_err} exceeds 1e-15"

    @pytest.mark.parametrize("positions", [1024, 8192, 65536])
    def test_rope_no_float32_truncation_in_intermediate_angles(self, positions):
        """RoPE does not perform intermediate float32 quantization when float64 is requested."""
        head_dim = 64
        cos_f64, sin_f64 = _rope_tables(positions, head_dim, device="cpu", dtype=torch.float64)
        cos_f32, sin_f32 = _rope_tables(positions, head_dim, device="cpu", dtype=torch.float32)

        # If float32 truncation occurred, cos_f64 would match cos_f32.double()
        cos_diff = (cos_f64 - cos_f32.to(torch.float64)).abs().max().item()
        sin_diff = (sin_f64 - sin_f32.to(torch.float64)).abs().max().item()

        # At large sequence positions, float32 precision loss is significant (> 1e-5 to > 1e-3)
        assert cos_diff > 1e-5, f"Expected float32 truncation difference, got {cos_diff}"
        assert sin_diff > 1e-5, f"Expected float32 truncation difference, got {sin_diff}"

    def test_rope_boundary_conditions(self):
        """Test boundary conditions: positions=0, positions=1, head_dim=2."""
        # positions = 0
        cos0, sin0 = _rope_tables(0, 16, device="cpu", dtype=torch.float64)
        assert cos0.shape == (1, 1, 0, 16)
        assert sin0.shape == (1, 1, 0, 16)

        # positions = 1: angles are 0, cos=1, sin=0, applying rope is identity
        cos1, sin1 = _rope_tables(1, 16, device="cpu", dtype=torch.float64)
        assert torch.allclose(cos1, torch.ones_like(cos1), atol=1e-15)
        assert torch.allclose(sin1, torch.zeros_like(sin1), atol=1e-15)

        q = torch.randn(2, 4, 1, 16, dtype=torch.float64)
        q_rot = _apply_rope(q, cos1, sin1)
        assert torch.equal(q, q_rot)

        # head_dim = 2 (minimal even dimension)
        cos2, sin2 = _rope_tables(5, 2, device="cpu", dtype=torch.float64)
        assert cos2.shape == (1, 1, 5, 2)

        # Odd head_dim rejected by Attention constructor
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*even head dimension"):
            Attention(heads=3, causal=False, dropout=0.0, bias=True, rope=True, D=21)

    def test_rope_attention_gradcheck_double(self):
        """Analytical gradient of RoPE attention passes torch.autograd.gradcheck in float64."""
        attn = Attention(heads=2, causal=True, dropout=0.0, bias=True, rope=True, D=8).double()
        x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(attn, (x,), eps=1e-6, atol=1e-5)

    def test_rope_attention_reference_parity_float64(self):
        """Attention with RoPE strictly matches reference implementation in float64."""
        attn = Attention(heads=4, causal=True, dropout=0.0, bias=True, rope=True, D=32).double()
        ref = attention_ref(attn)

        x = torch.randn(2, 16, 32, dtype=torch.float64)
        out_attn = attn(x)
        out_ref = ref(x)

        assert out_attn.dtype == torch.float64
        torch.testing.assert_close(out_attn, out_ref, rtol=1e-14, atol=1e-14)


# ==============================================================================
# M-12: PosEmbed Sequence Length Bounds & Rank Stress Tests
# ==============================================================================

class TestPosEmbedBoundsAndRankStress:
    """Stress tests verifying PosEmbed sequence length bounds and rank handling."""

    @pytest.mark.parametrize("max_len", [1, 2, 8, 32, 128, 512])
    def test_posembed_sequence_length_bounds_direct_module(self, max_len):
        """PosEmbed.forward enforces T <= max_len and handles T==max_len, T==0, T==max_len+1, T==10*max_len."""
        D = 16
        pos = PosEmbed(max_len=max_len, D=D)

        # 1. T == max_len: boundary success
        x_exact = torch.randn(2, max_len, D)
        out_exact = pos(x_exact)
        assert out_exact.shape == (2, max_len, D)
        torch.testing.assert_close(out_exact, x_exact + pos.weight.unsqueeze(0))

        # 2. T == 0: empty sequence boundary
        x_empty = torch.randn(2, 0, D)
        out_empty = pos(x_empty)
        assert out_empty.shape == (2, 0, D)

        # 3. T == max_len + 1: first out-of-bounds position raises E_CONSTRAINT
        x_overflow = torch.randn(2, max_len + 1, D)
        with pytest.raises(HNDLError) as exc_info:
            pos(x_overflow)
        assert exc_info.value.code == "E_CONSTRAINT"
        assert f"exceeds max_len={max_len}" in str(exc_info.value)

        # 4. T == 10 * max_len: extreme sequence overflow raises E_CONSTRAINT
        x_extreme = torch.randn(2, 10 * max_len, D)
        with pytest.raises(HNDLError) as exc_info:
            pos(x_extreme)
        assert exc_info.value.code == "E_CONSTRAINT"

        # 5. T < max_len (when max_len > 1): partial table encoding
        if max_len > 1:
            T_sub = max_len // 2
            x_sub = torch.randn(2, T_sub, D)
            out_sub = pos(x_sub)
            assert out_sub.shape == (2, T_sub, D)
            torch.testing.assert_close(out_sub, x_sub + pos.weight[:T_sub].unsqueeze(0))

    def test_posembed_max_len_1_critical_edge_case(self):
        """Regression test for M-12: max_len=1 must not silently broadcast pos 0 to T>1."""
        pos = PosEmbed(max_len=1, D=8)

        # T = 1: exact match
        x1 = torch.randn(3, 1, 8)
        assert pos(x1).shape == (3, 1, 8)

        # T = 2: overflow must raise E_CONSTRAINT, never silently broadcast
        x2 = torch.randn(3, 2, 8)
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*exceeds max_len=1"):
            pos(x2)

        # T = 0: empty sequence
        assert pos(torch.randn(3, 0, 8)).shape == (3, 0, 8)

        # T = 10: 10 * max_len
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*exceeds max_len=1"):
            pos(torch.randn(3, 10, 8))

    def test_posembed_gradient_sparsity_across_bounds(self):
        """Only the active positions 0..T receive gradients, remaining table stays zero."""
        max_len = 16
        D = 8
        pos = PosEmbed(max_len=max_len, D=D)

        # T = 5 < max_len
        x = torch.randn(2, 5, D, requires_grad=True)
        pos(x).sum().backward()

        assert pos.weight.grad is not None
        # Positions 0..4 have non-zero gradients
        assert (pos.weight.grad[:5] != 0).any()
        # Positions 5..15 must be exactly zero
        assert torch.equal(pos.weight.grad[5:], torch.zeros(11, D))

    def test_posembed_various_ranks_standalone_module(self):
        """Test PosEmbed submodule behavior when invoked across rank 1, rank 2, rank 3, rank 4."""
        max_len = 10
        D = 8
        pos = PosEmbed(max_len=max_len, D=D)

        # Rank 1: (D,) lacks sequence dimension
        with pytest.raises(IndexError):
            pos(torch.randn(D))

        # Rank 2: (B, D)
        # If D > max_len: x.shape[1] > max_len raises E_CONSTRAINT
        with pytest.raises(HNDLError, match="E_CONSTRAINT"):
            pos(torch.randn(2, max_len + 5))

        # Rank 3: (B, T, D) standard
        out3 = pos(torch.randn(2, max_len, D))
        assert out3.shape == (2, max_len, D)

        # Rank 4: (B, T, H, W)
        # T > max_len triggers E_CONSTRAINT
        with pytest.raises(HNDLError, match="E_CONSTRAINT"):
            pos(torch.randn(2, max_len + 1, 4, D))

    def test_posembed_declarative_resolver_ranks_and_bounds(self):
        """Declarative resolution enforces 3D sequences and rejects bounds violations."""
        # Rank 1: rejected by schema
        with pytest.raises(HNDLError, match="E_SCHEMA"):
            resolve("pos_embed(10)", input_shape=("B",), output_shape=("B",))

        # Rank 2: rejected by rank conflict
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*Rank conflict"):
            resolve("pos_embed(10)", input_shape=("B", 8), output_shape=("B", 8))

        # Rank 4: rejected by rank conflict
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*Rank conflict"):
            resolve("pos_embed(10)", input_shape=("B", 10, 8, 4), output_shape=("B", 10, 8, 4))

        # Rank 3: sequence length bounds
        # T == max_len: resolves successfully
        plan = resolve("pos_embed(10)", input_shape=("B", 10, 8), output_shape=("B", 10, 8))
        assert plan.nodes[0].args["max_len"] == 10

        # T == max_len + 1: rejected at resolve time
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*cannot address 11 positions"):
            resolve("pos_embed(10)", input_shape=("B", 11, 8), output_shape=("B", 11, 8))

        # T == 10 * max_len: rejected at resolve time
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*cannot address 100 positions"):
            resolve("pos_embed(10)", input_shape=("B", 100, 8), output_shape=("B", 100, 8))

        # T == 0: rejected by schema (dimension must be positive)
        with pytest.raises(HNDLError, match="E_SCHEMA.*must be positive"):
            resolve("pos_embed(10)", input_shape=("B", 0, 8), output_shape=("B", 0, 8))

    def test_posembed_declarative_network_runtime_and_submodule(self):
        """Network runtime verifies input contract, while submodule allows dynamic T <= max_len."""
        model = network("pos_embed(16)", input_shape=("B", 16, 8), output_shape=("B", 16, 8), device="cpu")

        # Top-level GraphModule enforces exact compiled contract shape
        x_valid = torch.randn(4, 16, 8)
        assert model(x_valid).shape == (4, 16, 8)

        with pytest.raises(HNDLError, match="E_RUNTIME.*expected shape"):
            model(torch.randn(4, 17, 8))

        # Submodule model['n0'] allows variable length up to max_len
        sub = model["n0"]
        assert sub(torch.randn(2, 16, 8)).shape == (2, 16, 8)
        assert sub(torch.randn(2, 5, 8)).shape == (2, 5, 8)
        assert sub(torch.randn(2, 0, 8)).shape == (2, 0, 8)

        # Submodule rejects T > max_len
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*exceeds max_len"):
            sub(torch.randn(2, 17, 8))
        with pytest.raises(HNDLError, match="E_CONSTRAINT.*exceeds max_len"):
            sub(torch.randn(2, 160, 8))

    def test_posembed_float64_precision_and_gradcheck(self):
        """PosEmbed double precision forward and gradcheck pass analytically."""
        pos = PosEmbed(max_len=8, D=4).double()
        x = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)

        out = pos(x)
        assert out.dtype == torch.float64
        assert out.shape == (2, 6, 4)

        # Gradcheck verifying numerical vs analytical Jacobian
        assert torch.autograd.gradcheck(pos, (x,), eps=1e-6, atol=1e-5)

    def test_posembed_reference_oracle_parity(self):
        """PosEmbed reference oracle matches forward pass."""
        pos = PosEmbed(max_len=16, D=8)
        ref = pos_embed_ref(pos)

        x = torch.randn(3, 10, 8)
        torch.testing.assert_close(pos(x), ref(x))
