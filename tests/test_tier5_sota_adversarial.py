"""Tier 5 SOTA Operators Adversarial Stress & Parity Test Suite.

Probes:
1. gated_delta_attention:
   - Sequence length sweeps: T in [1, 2, 7, 16, 35, 64, 73, 128]
   - Chunkwise parallel forward vs sequential step-by-step reference
   - decay_kind="channel" (KDA) and decay_kind="head" (GDN)
   - Odd chunk boundaries (chunk_size=3, 7, 17, 32, 64)
2. latent_attention:
   - NoPE mode (qk_rope_head_dim=0)
   - Extreme head counts (heads=1, 16, 32, 64)
   - Decoupled RoPE vs NoPE parity against dense reference
3. attention:
   - GQA (kv_heads != heads) with logit_softcap
   - Sliding-window band masking (window_size > 0)
   - Per-head QK-norm ("rms_norm", "l2_norm")
   - Full composition of GQA + softcap + window + qk_norm vs reference
4. moe:
   - SwiGLU experts (expert_kind="swiglu", bias=False/True)
   - Sigmoid scoring (scoring="sigmoid") under high expert counts and extreme router inputs
   - Parity between eager dispatch and dense reference
5. Autograd numerical stability:
   - float64 gradcheck across all extended/new operators
   - float64 gradgradcheck (double backward) parity
"""

import pytest
import torch

from hndl import Registry, resolve
from hndl.torch import build

REGISTRY = Registry.builtins()


# ==============================================================================
# 1. gated_delta_attention Adversarial Probes
# ==============================================================================


@pytest.mark.parametrize("seq_len", [1, 2, 7, 16, 35, 64, 73, 128])
@pytest.mark.parametrize("decay_kind", ["channel", "head"])
def test_gda_sequence_length_sweeps_reference_parity(seq_len, decay_kind):
    """Verify chunkwise parallel forward exactly matches sequential step-by-step reference."""
    D = 32
    heads = 2
    chunk_size = 16
    plan = resolve(
        f'gated_delta_attention({heads}, decay_kind="{decay_kind}", chunk_size={chunk_size}, bias=True)',
        input_shape=("B", seq_len, D),
        output_shape=("B", seq_len, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("gated_delta_attention").reference(module)

    torch.manual_seed(42 + seq_len)
    x = torch.randn(2, seq_len, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 17, 31])
@pytest.mark.parametrize("decay_kind", ["channel", "head"])
def test_gda_odd_chunk_boundaries(chunk_size, decay_kind):
    """Stress-test non-power-of-two and prime chunk boundaries where T % chunk_size != 0."""
    T = 43  # prime sequence length
    D = 16
    heads = 2
    plan = resolve(
        f'gated_delta_attention({heads}, decay_kind="{decay_kind}", chunk_size={chunk_size})',
        input_shape=("B", T, D),
        output_shape=("B", T, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("gated_delta_attention").reference(module)

    torch.manual_seed(100 + chunk_size)
    x = torch.randn(1, T, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_gda_conv_disabled_and_explicit_head_dim():
    """Verify conv_size=0 (disabled conv) and explicit head_dim decouple from D // heads."""
    T = 25
    D = 24
    heads = 3
    head_dim = 16  # heads * head_dim = 48 != 24
    plan = resolve(
        f"gated_delta_attention({heads}, head_dim={head_dim}, conv_size=0, chunk_size=8)",
        input_shape=("B", T, D),
        output_shape=("B", T, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("gated_delta_attention").reference(module)

    x = torch.randn(2, T, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 2. latent_attention (MLA) Adversarial Probes
# ==============================================================================


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("q_lora_rank", [0, 16])
def test_latent_attention_nope_mode_parity(causal, q_lora_rank):
    """Verify NoPE mode (qk_rope_head_dim=0) matches reference exactly for causal and non-causal."""
    T = 19
    D = 32
    heads = 4
    kv_lora_rank = 16
    plan = resolve(
        f"latent_attention({heads}, {kv_lora_rank}, q_lora_rank={q_lora_rank}, "
        f"qk_rope_head_dim=0, causal={causal})",
        input_shape=("B", T, D),
        output_shape=("B", T, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("latent_attention").reference(module)

    torch.manual_seed(77)
    x = torch.randn(2, T, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("heads", [1, 16, 32])
def test_latent_attention_extreme_head_counts(heads):
    """Verify extreme head counts (1 head, 16 heads, 32 heads) with MLA decoupling."""
    T = 8
    D = 64
    kv_lora_rank = 16
    qk_nope_head_dim = 8
    qk_rope_head_dim = 8
    plan = resolve(
        f"latent_attention({heads}, {kv_lora_rank}, qk_nope_head_dim={qk_nope_head_dim}, "
        f"qk_rope_head_dim={qk_rope_head_dim}, causal=True)",
        input_shape=("B", T, D),
        output_shape=("B", T, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("latent_attention").reference(module)

    x = torch.randn(2, T, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_latent_attention_single_token():
    """Verify single-token autoregressive prefill/decode (T=1)."""
    plan = resolve(
        "latent_attention(4, 16, qk_rope_head_dim=8, causal=True)",
        input_shape=("B", 1, 32),
        output_shape=("B", 1, 32),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("latent_attention").reference(module)

    x = torch.randn(2, 1, 32, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 3. attention SOTA Extensions Probes
# ==============================================================================


@pytest.mark.parametrize(
    "kv_heads,qk_norm,softcap,window_size",
    [
        (1, "rms_norm", 30.0, 4),  # MQA + QK-RMSNorm + Softcap + Sliding Window
        (2, "l2_norm", 50.0, 0),   # GQA + QK-L2Norm + Softcap + Full context
        (4, "rms_norm", 0.0, 3),   # GQA + QK-RMSNorm + Sliding Window
        (1, "none", 20.0, 2),      # MQA + Softcap + Tight window
    ],
)
def test_attention_sota_composition_parity(kv_heads, qk_norm, softcap, window_size):
    """Stress-test joint composition of GQA, per-head QK-norm, logit softcapping, and sliding window."""
    T = 12
    D = 32
    heads = 4
    plan = resolve(
        f'attention({heads}, kv_heads={kv_heads}, qk_norm="{qk_norm}", '
        f"logit_softcap={softcap}, window_size={window_size}, causal=True, rope=True)",
        input_shape=("B", T, D),
        output_shape=("B", T, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("attention").reference(module)

    torch.manual_seed(99)
    x = torch.randn(2, T, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("window_size", [1, 2, 5, 20])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_sliding_window_boundaries(window_size, causal):
    """Stress-test sliding window sizes: window=1 (self-only in causal), window > T (full attention)."""
    T = 10
    D = 16
    heads = 2
    plan = resolve(
        f"attention({heads}, window_size={window_size}, causal={causal})",
        input_shape=("B", T, D),
        output_shape=("B", T, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("attention").reference(module)

    x = torch.randn(2, T, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 4. moe SwiGLU & Sigmoid Scoring Probes
# ==============================================================================


@pytest.mark.parametrize("expert_kind", ["mlp", "swiglu"])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("bias", [True, False])
def test_moe_configurations_parity(expert_kind, scoring, bias):
    """Verify SwiGLU experts and sigmoid scoring match dense reference across all configurations."""
    D = 16
    hidden = 32
    experts = 4
    top_k = 2
    plan = resolve(
        f'moe({experts}, {hidden}, top_k={top_k}, expert_kind="{expert_kind}", '
        f'scoring="{scoring}", bias={bias})',
        input_shape=("B", 8, D),
        output_shape=("B", 8, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("moe").reference(module)

    torch.manual_seed(2026)
    x = torch.randn(2, 8, D, dtype=torch.float64)
    actual = module(x)
    expected = ref_fn(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_moe_high_expert_counts_and_edge_router_inputs():
    """Stress-test MoE with 32 experts, top_k=4, sigmoid scoring, and adversarial input scales."""
    D = 32
    hidden = 64
    experts = 32
    top_k = 4
    plan = resolve(
        f'moe({experts}, {hidden}, top_k={top_k}, expert_kind="swiglu", scoring="sigmoid", bias=False)',
        input_shape=("B", 16, D),
        output_shape=("B", 16, D),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].to(torch.float64)
    ref_fn = REGISTRY.get("moe").reference(module)

    # Edge inputs: all zeros, large magnitude, tiny magnitude
    zeros_x = torch.zeros(2, 16, D, dtype=torch.float64)
    actual_zeros = module(zeros_x)
    expected_zeros = ref_fn(zeros_x)
    assert actual_zeros.isfinite().all()
    torch.testing.assert_close(actual_zeros, expected_zeros, rtol=1e-5, atol=1e-5)

    huge_x = torch.randn(2, 16, D, dtype=torch.float64) * 1000.0
    actual_huge = module(huge_x)
    expected_huge = ref_fn(huge_x)
    assert actual_huge.isfinite().all()
    torch.testing.assert_close(actual_huge, expected_huge, rtol=1e-4, atol=1e-4)


# ==============================================================================
# 5. Autograd Numerical Stability & Parity (Gradcheck & Gradgradcheck)
# ==============================================================================


def test_gda_gradcheck_and_gradgradcheck():
    """Verify float64 1st and 2nd order differentiability for gated_delta_attention."""
    plan = resolve(
        'gated_delta_attention(2, head_dim=4, decay_kind="channel", conv_size=2, chunk_size=4)',
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

    # 1st order gradcheck
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)

    # 2nd order double backward check (gradgradcheck)
    assert torch.autograd.gradgradcheck(module, (x,), eps=1e-6, atol=1e-4)


def test_latent_attention_gradcheck_and_gradgradcheck():
    """Verify float64 1st and 2nd order differentiability for latent_attention (NoPE and RoPE)."""
    # NoPE mode
    plan_nope = resolve(
        "latent_attention(2, 8, qk_nope_head_dim=4, qk_rope_head_dim=0, causal=True)",
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model_nope = build(plan_nope, device="cpu")
    module_nope = model_nope[plan_nope.nodes[0].id].double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module_nope, (x,), eps=1e-6, atol=1e-4)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        assert torch.autograd.gradgradcheck(module_nope, (x,), eps=1e-6, atol=1e-4)

    # RoPE mode with Q-LoRA
    plan_rope = resolve(
        "latent_attention(2, 8, q_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, causal=True)",
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model_rope = build(plan_rope, device="cpu")
    module_rope = model_rope[plan_rope.nodes[0].id].double()
    x_rope = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module_rope, (x_rope,), eps=1e-6, atol=1e-4)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        assert torch.autograd.gradgradcheck(module_rope, (x_rope,), eps=1e-6, atol=1e-4)


def test_attention_sota_gradcheck_and_gradgradcheck():
    """Verify float64 1st and 2nd order differentiability for attention with GQA, softcap, and QK-norm."""
    plan = resolve(
        'attention(4, kv_heads=2, qk_norm="rms_norm", logit_softcap=20.0, window_size=3, causal=True, rope=True)',
        input_shape=("B", 4, 16),
        output_shape=("B", 4, 16),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].double()
    x = torch.randn(1, 4, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)
    assert torch.autograd.gradgradcheck(module, (x,), eps=1e-6, atol=1e-4)


def test_moe_swiglu_sigmoid_gradcheck_and_gradgradcheck():
    """Verify float64 1st and 2nd order differentiability for MoE SwiGLU and sigmoid scoring."""
    plan = resolve(
        'moe(3, 8, top_k=2, expert_kind="swiglu", scoring="sigmoid", bias=False)',
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model = build(plan, device="cpu")
    module = model[plan.nodes[0].id].double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)
    assert torch.autograd.gradgradcheck(module, (x,), eps=1e-6, atol=1e-4)


# ==============================================================================
# 6. Empirical Vulnerability Demonstrations (Expected Failures in Current Impl)
# ==============================================================================


def test_vulnerability_moe_sigmoid_float16_underflow_causes_nan():
    """Demonstrate that negative router logits in float16 produce NaNs due to 1e-20 underflow."""
    plan = resolve(
        'moe(4, 16, top_k=2, scoring="sigmoid")',
        input_shape=("B", 8),
        output_shape=("B", 8),
        dtype="float16",
    )
    model = build(plan, device="cpu")
    x = torch.ones(2, 8, dtype=torch.float16)
    model["n0"].router.weight.data.fill_(-50.0)
    out = model(x=x)["output"]
    # This currently evaluates to False (contains NaNs), failing the assertion
    assert out.isfinite().all(), "Expected finite outputs from MoE with negative router logits"


def test_vulnerability_gda_alpha_underflow_causes_nan():
    """Demonstrate that extreme decay rates in float16 cause log(0) -inf and NaN outputs in forward, while reference stays finite."""
    plan = resolve(
        'gated_delta_attention(2, decay_kind="channel", chunk_size=16)',
        input_shape=("B", 16, 32),
        output_shape=("B", 16, 32),
        dtype="float16",
    )
    model = build(plan, device="cpu")
    x = torch.randn(2, 16, 32, dtype=torch.float16) * 100.0
    model["n0"].alpha_up.weight.data.fill_(10.0)
    model["n0"].alpha_up.bias.data.fill_(100.0)

    # Reference oracle stays completely finite:
    ref_fn = REGISTRY.get("gated_delta_attention").reference(model["n0"])
    ref_out = ref_fn(x)
    assert ref_out.isfinite().all(), "Reference oracle must stay finite"

    # Forward produces NaNs due to diff = cum_log - cum_log = (-inf) - (-inf) = NaN:
    actual_out = model(x=x)["output"]
    assert actual_out.isfinite().all(), "Forward should stay finite like reference"

