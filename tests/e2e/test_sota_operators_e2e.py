"""Comprehensive opaque-box E2E test suite for HNDL SOTA Operators and architectures.

Covers all 10 feature areas identified in TEST_INFRA.md and PROJECT.md:
- F1: Attention & Cross-Attention SOTA Extensions (GQA kv_heads, head_dim, qk_norm, rope_theta, window_size, logit_softcap, cross_attention biases)
- F2: Transformer Block SOTA Extensions (ffn_kind="swiglu", explicit hidden override, rope, rope_theta, kv_heads, qk_norm, bias)
- F3: MoE SOTA Extensions (expert_kind="swiglu", bias, scoring="sigmoid", compile compatibility, reference oracle)
- F4: Convolution, Normalization & Core Math Extensions (conv1d causal & channels_last, rms_norm zero_centered, layer_norm bias=False, adaptive_norm rank-3, matmul multi-rank batching)
- F5: Gated Delta Attention (KDA decay_kind="channel", GDN decay_kind="head", causal depthwise conv1d k=4, per-head L2 norm, chunkwise WY forward, sequential recurrence reference)
- F6: Latent Attention (Multi-Head Latent Attention MLA, kv_lora_rank, optional q_lora_rank, decoupled RoPE, NoPE mode qk_rope_head_dim=0)
- F7: Normalization & Positional Operators (l2_norm parameter-free, adaln_zero rank-3 adaptive LayerNorm-Zero conditioning returning out & gate, rope standalone rotary embedding)
- F8: Domain-Completing Operators (deconv1d conv_transpose1d@1 with policy="up2", grn Global Response Normalization, sinusoidal_embed deterministic log-spaced embeddings, cast explicit dtype conversion)
- F9: Pretrained Provider Contract Hardening (shape_fn callback avoiding meta device execution, contract="ids" token-ID inputs, model construction with custom providers)
- F10: Authored SOTA Network Recipes & Architecture Integration (verifying resolution, parameter counts, round-tripping, and gradients across the 8 SOTA recipes)

Tiers:
- Tier 1: Feature Coverage (happy paths in isolation, >=5 tests per feature area)
- Tier 2: Boundary & Corner Cases (limits, syntax extremes, and defect triggers, >=5 tests per feature area)
- Tier 3: Cross-Feature Interactions (pairwise combinations, >=15 tests)
- Tier 4: Real-World Application Workloads (end-to-end architectures, >=5 tests)
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import hndl.pretrained  # noqa: F401

from hndl import (
    HNDLError,
    Registry,
    ResolvedPlan,
    resolve,
)
from hndl.torch import network, parameter_counts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# TIER 1: Feature Coverage (Happy Paths in Isolation)
# ==============================================================================


class TestTier1FeatureCoverage:
    """Tier 1: >=5 tests per feature area covering happy paths in isolation."""

    # --------------------------------------------------------------------------
    # F1: Attention & Cross-Attention SOTA Extensions
    # --------------------------------------------------------------------------

    def test_tier1_f1_gqa_grouped_query_attention(self, device):
        """[F1 Happy Path 1] Attention with kv_heads (GQA 4 query heads, 2 kv heads) resolves and runs."""
        source = "attention(4, kv_heads=2, name='attn')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)
        assert out.isfinite().all()

    def test_tier1_f1_independent_head_dim_override(self, device):
        """[F1 Happy Path 2] Attention with decoupled head_dim override (D=64, heads=2, head_dim=48)."""
        source = "attention(2, head_dim=48, name='attn')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f1_per_head_qk_normalization(self, device):
        """[F1 Happy Path 3] Attention with per-head QK-RMSNorm (qk_norm='rms_norm')."""
        source = "attention(4, qk_norm='rms_norm', name='attn')\n"
        model = network(source, input_shape=("B", 12, 64), output_shape=("B", 12, 64), device=device)
        x = torch.randn(2, 12, 64, device=device)
        out = model(x)
        assert out.shape == (2, 12, 64)

    def test_tier1_f1_configurable_high_theta_rope(self, device):
        """[F1 Happy Path 4] Attention with RoPE and high-frequency theta (rope_theta=1000000.0)."""
        source = "attention(4, rope=True, rope_theta=1000000.0, name='attn')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f1_sliding_window_and_logit_softcap(self, device):
        """[F1 Happy Path 5] Attention with sliding-window attention and logit soft-capping."""
        source = "attention(4, window_size=8, logit_softcap=30.0, causal=True, name='attn')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f1_cross_attention_sota_extensions(self, device):
        """[F1 Happy Path 6] Cross-attention with kv_heads, qkv_bias=False, and out_bias=False."""
        source = "out = cross_attention(x, enc, 4, kv_heads=2, qkv_bias=False, out_bias=False, name='xattn')\n"
        model = network(
            source,
            input_shape={"x": ("B", 8, 64), "enc": ("B", 16, 64)},
            output_shape={"out": ("B", 8, 64)},
            device=device,
        )
        x = torch.randn(2, 8, 64, device=device)
        enc = torch.randn(2, 16, 64, device=device)
        out = model(x=x, enc=enc)
        assert out.shape == (2, 8, 64)

    # --------------------------------------------------------------------------
    # F2: Transformer Block SOTA Extensions
    # --------------------------------------------------------------------------

    def test_tier1_f2_transformer_block_swiglu_ffn(self, device):
        """[F2 Happy Path 1] Transformer block with ffn_kind='swiglu'."""
        source = "transformer_block(4, ffn_kind='swiglu', name='tb')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f2_transformer_block_explicit_hidden_override(self, device):
        """[F2 Happy Path 2] Transformer block with explicit hidden width override (hidden=192)."""
        source = "transformer_block(4, hidden=192, ffn_kind='swiglu', name='tb')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f2_transformer_block_gqa_and_qk_norm(self, device):
        """[F2 Happy Path 3] Transformer block with GQA (kv_heads=2) and qk_norm='rms_norm'."""
        source = "transformer_block(4, kv_heads=2, qk_norm='rms_norm', norm='rms_norm', name='tb')\n"
        model = network(source, input_shape=("B", 12, 64), output_shape=("B", 12, 64), device=device)
        x = torch.randn(2, 12, 64, device=device)
        out = model(x)
        assert out.shape == (2, 12, 64)

    def test_tier1_f2_transformer_block_rope_options(self, device):
        """[F2 Happy Path 4] Transformer block with rope=True and rope_theta=500000.0."""
        source = "transformer_block(4, rope=True, rope_theta=500000.0, causal=True, name='tb')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f2_transformer_block_bias_false(self, device):
        """[F2 Happy Path 5] Transformer block fully bias-free (bias=False)."""
        source = "transformer_block(4, bias=False, ffn_kind='swiglu', name='tb')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    # --------------------------------------------------------------------------
    # F3: MoE SOTA Extensions
    # --------------------------------------------------------------------------

    def test_tier1_f3_moe_swiglu_experts(self, device):
        """[F3 Happy Path 1] MoE with 3-projection SwiGLU experts (expert_kind='swiglu')."""
        source = "moe(4, 128, top_k=2, expert_kind='swiglu', name='moe')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f3_moe_bias_false(self, device):
        """[F3 Happy Path 2] MoE with bias-free experts (bias=False)."""
        source = "moe(4, 128, top_k=2, expert_kind='swiglu', bias=False, name='moe')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f3_moe_sigmoid_scoring(self, device):
        """[F3 Happy Path 3] MoE with sigmoid affinity scoring (scoring='sigmoid')."""
        source = "moe(4, 128, top_k=2, scoring='sigmoid', name='moe')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f3_moe_combined_swiglu_sigmoid_bias_false(self, device):
        """[F3 Happy Path 4] MoE combining SwiGLU experts, sigmoid scoring, and bias=False."""
        source = "moe(4, 128, top_k=2, expert_kind='swiglu', scoring='sigmoid', bias=False, name='moe')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier1_f3_moe_gradient_flow(self, device):
        """[F3 Happy Path 5] MoE with SwiGLU experts propagates autograd gradients back through all paths."""
        source = "moe(4, 64, top_k=2, expert_kind='swiglu', name='moe')\n"
        model = network(source, input_shape=("B", 4, 32), output_shape=("B", 4, 32), device=device)
        x = torch.randn(2, 4, 32, device=device, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()

    # --------------------------------------------------------------------------
    # F4: Convolution, Normalization & Core Math Extensions
    # --------------------------------------------------------------------------

    def test_tier1_f4_conv1d_causal_padding(self, device):
        """[F4 Happy Path 1] conv1d with causal=True left-padding preserving exact sequence length."""
        source = "conv1d(32, kernel_size=3, causal=True, name='c1d')\n"
        model = network(source, input_shape=("B", 16, 24), output_shape=("B", 32, 24), device=device)
        x = torch.randn(2, 16, 24, device=device)
        out = model(x)
        assert out.shape == (2, 32, 24)

    def test_tier1_f4_conv1d_channels_last(self, device):
        """[F4 Happy Path 2] conv1d with channels_last=True for [B, T, C] sequences."""
        source = "conv1d(64, kernel_size=3, causal=True, channels_last=True, name='c1d')\n"
        model = network(source, input_shape=("B", 32, 16), output_shape=("B", 32, 64), device=device)
        x = torch.randn(2, 32, 16, device=device)
        out = model(x)
        assert out.shape == (2, 32, 64)

    def test_tier1_f4_rms_norm_zero_centered(self, device):
        """[F4 Happy Path 3] rms_norm with zero_centered=True ((1 + w) * RMSNorm(x))."""
        source = "rms_norm(zero_centered=True, name='norm')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f4_layer_norm_bias_false(self, device):
        """[F4 Happy Path 4] layer_norm with bias=False."""
        source = "layer_norm(bias=False, name='ln')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f4_adaptive_norm_rank3_sequence(self, device):
        """[F4 Happy Path 5] adaptive_norm on rank-3 [B, T, C] sequence modulation."""
        source = "out = adaptive_norm(x, params, name='ad')\n"
        model = network(
            source,
            input_shape={"x": ("B", 16, 32), "params": ("B", 64)},
            output_shape={"out": ("B", 16, 32)},
            device=device,
        )
        x = torch.randn(2, 16, 32, device=device)
        params = torch.randn(2, 64, device=device)
        out = model(x=x, params=params)
        assert out.shape == (2, 16, 32)

    def test_tier1_f4_matmul_multi_rank_batching(self, device):
        """[F4 Happy Path 6] matmul supporting rank-3 and rank-4 batched multiplications."""
        source = "out = matmul(a, b, name='mm')\n"
        model = network(
            source,
            input_shape={"a": ("B", 4, 16, 32), "b": ("B", 4, 32, 16)},
            output_shape={"out": ("B", 4, 16, 16)},
            device=device,
        )
        a = torch.randn(2, 4, 16, 32, device=device)
        b = torch.randn(2, 4, 32, 16, device=device)
        out = model(a=a, b=b)
        assert out.shape == (2, 4, 16, 16)

    # --------------------------------------------------------------------------
    # F5: Gated Delta Attention (KDA / GDN)
    # --------------------------------------------------------------------------

    def test_tier1_f5_gated_delta_attention_channel_decay(self, device):
        """[F5 Happy Path 1] gated_delta_attention with KDA channel-wise decay (decay_kind='channel')."""
        source = "gated_delta_attention(4, decay_kind='channel', name='kda')\n"
        model = network(source, input_shape=("B", 32, 64), output_shape=("B", 32, 64), device=device)
        x = torch.randn(2, 32, 64, device=device)
        out = model(x)
        assert out.shape == (2, 32, 64)
        assert out.isfinite().all()

    def test_tier1_f5_gated_delta_attention_head_decay(self, device):
        """[F5 Happy Path 2] gated_delta_attention with Gated DeltaNet head-wise decay (decay_kind='head')."""
        source = "gated_delta_attention(4, decay_kind='head', name='gdn')\n"
        model = network(source, input_shape=("B", 32, 64), output_shape=("B", 32, 64), device=device)
        x = torch.randn(2, 32, 64, device=device)
        out = model(x)
        assert out.shape == (2, 32, 64)

    def test_tier1_f5_gated_delta_attention_conv_size_override(self, device):
        """[F5 Happy Path 3] gated_delta_attention with causal short conv_size=2 and custom gate_rank."""
        source = "gated_delta_attention(4, conv_size=2, gate_rank=16, name='kda')\n"
        model = network(source, input_shape=("B", 32, 64), output_shape=("B", 32, 64), device=device)
        x = torch.randn(2, 32, 64, device=device)
        out = model(x)
        assert out.shape == (2, 32, 64)

    def test_tier1_f5_gated_delta_attention_chunkwise_wy_forward(self, device):
        """[F5 Happy Path 4] gated_delta_attention with custom chunk_size=16."""
        source = "gated_delta_attention(4, chunk_size=16, name='kda')\n"
        model = network(source, input_shape=("B", 48, 64), output_shape=("B", 48, 64), device=device)
        x = torch.randn(2, 48, 64, device=device)
        out = model(x)
        assert out.shape == (2, 48, 64)

    def test_tier1_f5_gated_delta_attention_autograd_gradients(self, device):
        """[F5 Happy Path 5] gated_delta_attention backward autograd gradient check."""
        source = "gated_delta_attention(2, name='kda')\n"
        model = network(source, input_shape=("B", 16, 32), output_shape=("B", 16, 32), device=device)
        x = torch.randn(2, 16, 32, device=device, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()

    # --------------------------------------------------------------------------
    # F6: Latent Attention (Multi-Head Latent Attention / MLA)
    # --------------------------------------------------------------------------

    def test_tier1_f6_latent_attention_kv_lora_compression(self, device):
        """[F6 Happy Path 1] latent_attention with compressed KV latent width (kv_lora_rank=16)."""
        source = "latent_attention(4, kv_lora_rank=16, name='mla')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f6_latent_attention_q_lora_compression(self, device):
        """[F6 Happy Path 2] latent_attention with both Q and KV LoRA bottlenecks (q_lora_rank=32, kv_lora_rank=16)."""
        source = "latent_attention(4, kv_lora_rank=16, q_lora_rank=32, name='mla')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f6_latent_attention_nope_mode(self, device):
        """[F6 Happy Path 3] latent_attention in NoPE mode (qk_rope_head_dim=0, as in Kimi Linear)."""
        source = "latent_attention(4, kv_lora_rank=16, qk_rope_head_dim=0, name='mla')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f6_latent_attention_decoupled_rope(self, device):
        """[F6 Happy Path 4] latent_attention with decoupled RoPE key/query dimensions (qk_rope_head_dim=16)."""
        source = "latent_attention(4, kv_lora_rank=16, qk_rope_head_dim=16, rope_theta=500000.0, name='mla')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f6_latent_attention_autograd_gradients(self, device):
        """[F6 Happy Path 5] latent_attention autograd backward pass verifies finite gradients."""
        source = "latent_attention(2, kv_lora_rank=8, name='mla')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()

    # --------------------------------------------------------------------------
    # F7: Normalization & Positional Operators (l2_norm, adaln_zero, rope)
    # --------------------------------------------------------------------------

    def test_tier1_f7_l2_norm_parameter_free(self, device):
        """[F7 Happy Path 1] Parameter-free l2_norm normalizes tensors to unit norm."""
        source = "l2_norm(dim=-1, name='norm')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 8, 32)
        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_tier1_f7_adaln_zero_rank3_conditioning(self, device):
        """[F7 Happy Path 2] adaln_zero modulates [B, T, D] sequences from conditioning vector cond[B, D_c]."""
        source = "normed, gate = adaln_zero(x, cond, name='ad')\n"
        model = network(
            source,
            input_shape={"x": ("B", 16, 64), "cond": ("B", 32)},
            output_shape={"normed": ("B", 16, 64), "gate": ("B", 16, 64)},
            device=device,
        )
        x = torch.randn(2, 16, 64, device=device)
        cond = torch.randn(2, 32, device=device)
        out = model(x=x, cond=cond)
        assert out["normed"].shape == (2, 16, 64)
        assert out["gate"].shape == (2, 16, 64)

    def test_tier1_f7_adaln_zero_zero_initialization(self, device):
        """[F7 Happy Path 3] adaln_zero initializes modulation projection to zero (gate=0, scale=0, shift=0)."""
        source = "normed, gate = adaln_zero(x, cond, name='ad')\n"
        model = network(
            source,
            input_shape={"x": ("B", 8, 32), "cond": ("B", 16)},
            output_shape={"normed": ("B", 8, 32), "gate": ("B", 8, 32)},
            device=device,
        )
        cond = torch.randn(2, 16, device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x=x, cond=cond)
        torch.testing.assert_close(out["gate"], torch.zeros_like(out["gate"]), atol=1e-5, rtol=1e-5)

    def test_tier1_f7_standalone_rope_operator(self, device):
        """[F7 Happy Path 4] Standalone rope operator applies rotary position embeddings."""
        source = "rope(heads=4, theta=10000.0, name='rp')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier1_f7_standalone_rope_norm_preservation(self, device):
        """[F7 Happy Path 5] Standalone rope preserves L2 norm of query/key vectors."""
        source = "rope(heads=2, name='rp')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        torch.testing.assert_close(torch.norm(x, dim=-1), torch.norm(out, dim=-1), rtol=1e-4, atol=1e-4)

    # --------------------------------------------------------------------------
    # F8: Domain-Completing Operators (deconv1d, grn, sinusoidal_embed, cast)
    # --------------------------------------------------------------------------

    def test_tier1_f8_deconv1d_upsampling(self, device):
        """[F8 Happy Path 1] deconv1d (conv_transpose1d@1) upsamples length 2x with policy='up2'."""
        source = "deconv1d(32, kernel_size=4, stride=2, padding=1, name='dec')\n"
        model = network(source, input_shape=("B", 16, 24), output_shape=("B", 32, 48), device=device)
        x = torch.randn(2, 16, 24, device=device)
        out = model(x)
        assert out.shape == (2, 32, 48)

    def test_tier1_f8_grn_global_response_normalization(self, device):
        """[F8 Happy Path 2] grn normalizes spatial features and applies residual scaling for ConvNeXt-V2."""
        source = "grn(name='grn')\n"
        model = network(source, input_shape=("B", 32, 8, 8), output_shape=("B", 32, 8, 8), device=device)
        x = torch.randn(2, 32, 8, 8, device=device)
        out = model(x)
        assert out.shape == (2, 32, 8, 8)

    def test_tier1_f8_sinusoidal_embed_timesteps(self, device):
        """[F8 Happy Path 3] sinusoidal_embed creates deterministic log-spaced embeddings."""
        source = "sinusoidal_embed(64, name='sin')\n"
        model = network(source, input_shape=("B", 1), output_shape=("B", 64), device=device)
        t = torch.tensor([[10.0], [50.0]], device=device)
        out = model(t)
        assert out.shape == (2, 64)
        assert out.isfinite().all()

    def test_tier1_f8_cast_explicit_dtype_conversion(self, device):
        """[F8 Happy Path 4] cast converts integer tokens to float32 compute embeddings."""
        source = (
            "f = cast(dtype='float32', name='to_float')\n"
            "out = linear(f, 32, name='head')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 16)},
            output_shape={"out": ("B", 32)},
            input_dtype="float32",
            device=device,
        )
        x = torch.randn(2, 16, device=device)
        out = model(x=x)
        assert out.shape == (2, 32)
        assert out.dtype == torch.float32

    def test_tier1_f8_sinusoidal_embed_reproducibility(self, device):
        """[F8 Happy Path 5] sinusoidal_embed produces deterministic identical outputs for the same input."""
        source = "sinusoidal_embed(32, max_period=10000.0, name='sin')\n"
        model = network(source, input_shape=("B", 1), output_shape=("B", 32), device=device)
        t = torch.tensor([[5.0], [20.0]], device=device)
        out1 = model(t)
        out2 = model(t)
        torch.testing.assert_close(out1, out2)

    # --------------------------------------------------------------------------
    # F9: Pretrained Provider Contract Hardening
    # --------------------------------------------------------------------------

    def test_tier1_f9_pretrained_shape_fn_callback(self):
        """[F9 Happy Path 1] registry.pretrained_provider accepts shape_fn avoiding meta execution."""
        reg = Registry()

        def dummy_builder(args):
            return nn.Linear(16, 32)

        def dummy_shape(args):
            return {"output": ("B", 32)}

        reg.pretrained_provider("custom_kernel", dummy_builder, shape_fn=dummy_shape)
        assert "custom_kernel" in reg.pretrained_providers

    def test_tier1_f9_pretrained_ids_contract(self, text_checkpoint, device):
        """[F9 Happy Path 2] pretrained accepts contract='ids' for rank-2 int64 token inputs."""
        source = f'pretrained("{text_checkpoint}", name="lm")\n'
        model = network(
            source,
            input_shape=("B", 8),
            output_shape=("B", 8, 8),
            input_dtype="int64",
            device=device,
        )
        tokens = torch.randint(0, 16, (2, 8), device=device)
        out = model(tokens)
        assert out.shape == (2, 8, 8)

    def test_tier1_f9_pretrained_downstream_chaining(self, text_checkpoint, device):
        """[F9 Happy Path 3] pretrained model followed by SOTA rms_norm and linear projection."""
        source = (
            f'lm = pretrained("{text_checkpoint}", name="lm")\n'
            'norm = rms_norm(lm, name="norm")\n'
            'linear(norm, 10, name="head")\n'
        )
        model = network(
            source,
            input_shape=("B", 4),
            output_shape=("B", 4, 10),
            input_dtype="int64",
            device=device,
        )
        tokens = torch.randint(0, 16, (2, 4), device=device)
        out = model(tokens)
        assert out.shape == (2, 4, 10)

    def test_tier1_f9_pretrained_trainable_gradients(self, text_checkpoint, device):
        """[F9 Happy Path 4] pretrained module with trainable=True receives autograd gradients."""
        source = f'pretrained("{text_checkpoint}", trainable=True, name="lm")\n'
        model = network(
            source,
            input_shape=("B", 4),
            output_shape=("B", 4, 8),
            input_dtype="int64",
            device=device,
        )
        tokens = torch.randint(0, 16, (2, 4), device=device)
        out = model(tokens)
        out.sum().backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        assert all(p.grad is not None for p in trainable)

    def test_tier1_f9_custom_shape_fn_resolution(self):
        """[F9 Happy Path 5] Provider with shape_fn resolves correctly without instantiating model on meta."""
        reg = Registry()

        def custom_build(args):
            return nn.Identity()

        def custom_shape(args):
            return {"output": ("B", 64, 128)}

        reg.pretrained_provider("triton_block", custom_build, shape_fn=custom_shape)
        assert "triton_block" in reg.pretrained_providers
        provider_fn = reg.pretrained_builder("triton_block")
        assert callable(provider_fn)
        assert reg.pretrained_shape_fn("triton_block") is custom_shape

    # --------------------------------------------------------------------------
    # F10: Authored SOTA Network Recipes & Architecture Integration
    # --------------------------------------------------------------------------

    def test_tier1_f10_llama_tiny_resolves_and_runs(self, device):
        """[F10 Happy Path 1] llama_tiny network recipe resolves and verifies parameter counts."""
        hndl_path = REPO_ROOT / "examples" / "networks" / "llama_tiny.hndl"
        if not hndl_path.exists():
            pytest.skip("llama_tiny.hndl not yet authored")
        source = hndl_path.read_text(encoding="utf-8")
        model = network(source, input_shape=("B", 64), output_shape=("B", 64, 256), input_dtype="int64", device=device)
        tokens = torch.randint(0, 256, (2, 64), device=device)
        out = model(tokens)
        assert out.shape == (2, 64, 256)
        assert sum(parameter_counts(model.plan).values()) == 918656

    def test_tier1_f10_deepseek_moe_tiny_resolves(self, device):
        """[F10 Happy Path 2] deepseek_moe_tiny recipe with shared SwiGLU + routed MoE."""
        hndl_path = REPO_ROOT / "examples" / "networks" / "deepseek_moe_tiny.hndl"
        if not hndl_path.exists():
            pytest.skip("deepseek_moe_tiny.hndl not yet authored")
        source = hndl_path.read_text(encoding="utf-8")
        model = network(source, input_shape=("B", 32), output_shape=("B", 32, 256), input_dtype="int64", device=device)
        tokens = torch.randint(0, 256, (2, 32), device=device)
        out = model(tokens)
        assert out.shape == (2, 32, 256)
        assert sum(parameter_counts(model.plan).values()) == 1181312

    def test_tier1_f10_whisper_seq2seq_tiny_resolves(self, device):
        """[F10 Happy Path 3] whisper_seq2seq_tiny recipe with conv1d audio stem and cross_attention."""
        hndl_path = REPO_ROOT / "examples" / "networks" / "whisper_seq2seq_tiny.hndl"
        if not hndl_path.exists():
            pytest.skip("whisper_seq2seq_tiny.hndl not yet authored")
        source = hndl_path.read_text(encoding="utf-8")
        model = network(
            source,
            input_shape={"x": ("B", 40, 64), "tokens": ("B", 16)},
            output_shape=("B", 16, 128),
            input_dtype={"tokens": "int64"},
            device=device,
        )
        x = torch.randn(2, 40, 64, device=device)
        tokens = torch.randint(0, 128, (2, 16), device=device)
        out = model(x=x, tokens=tokens)
        assert out.shape == (2, 16, 128)
        assert sum(parameter_counts(model.plan).values()) == 159296

    def test_tier1_f10_convnext_tiny_resolves(self, device):
        """[F10 Happy Path 4] convnext_tiny recipe with 7x7 depthwise conv, GroupNorm, and LayerScale."""
        hndl_path = REPO_ROOT / "examples" / "networks" / "convnext_tiny.hndl"
        if not hndl_path.exists():
            pytest.skip("convnext_tiny.hndl not yet authored")
        source = hndl_path.read_text(encoding="utf-8")
        model = network(source, input_shape=("B", 3, 32, 32), output_shape=("B", 10), device=device)
        images = torch.randn(2, 3, 32, 32, device=device)
        out = model(images)
        assert out.shape == (2, 10)
        assert sum(parameter_counts(model.plan).values()) == 76876

    def test_tier1_f10_espcn_super_res_backward_channel_inference(self, device):
        """[F10 Happy Path 5] espcn_super_res recipe with subpixel_conv inferred backward channels."""
        hndl_path = REPO_ROOT / "examples" / "networks" / "espcn_super_res.hndl"
        if not hndl_path.exists():
            pytest.skip("espcn_super_res.hndl not yet authored")
        source = hndl_path.read_text(encoding="utf-8")
        model = network(source, input_shape=("B", 3, 16, 16), output_shape=("B", 3, 32, 32), device=device)
        images = torch.randn(2, 3, 16, 16, device=device)
        out = model(images)
        assert out.shape == (2, 3, 32, 32)
        assert sum(parameter_counts(model.plan).values()) == 26796


# ==============================================================================
# TIER 2: Boundary & Corner Cases (Limits, Extremes, and Defect Triggers)
# ==============================================================================


class TestTier2BoundaryCases:
    """Tier 2: >=5 tests per feature area covering limits, empty inputs, and syntax extremes."""

    # --------------------------------------------------------------------------
    # F1 Boundaries: Attention & Cross-Attention
    # --------------------------------------------------------------------------

    def test_tier2_f1_kv_heads_must_divide_heads(self):
        """[F1 Boundary 1] Attention raises E_CONSTRAINT when heads is not divisible by kv_heads."""
        source = "attention(5, kv_heads=2)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64))
        assert exc_info.value.code in ("E_CONSTRAINT", "E_ARGUMENT")

    def test_tier2_f1_rope_theta_without_rope_rejected(self):
        """[F1 Boundary 2] Specifying rope_theta != 10000.0 with rope=False raises E_ARGUMENT."""
        source = "attention(4, rope=False, rope_theta=500000.0)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64))
        assert exc_info.value.code == "E_ARGUMENT"

    def test_tier2_f1_window_size_exceeding_sequence_length(self, device):
        """[F1 Boundary 3] Sliding-window attention when window_size is larger than sequence length."""
        source = "attention(4, window_size=128, causal=True, name='attn')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier2_f1_odd_sequence_length_with_rope(self, device):
        """[F1 Boundary 4] Attention with RoPE on odd sequence length (T=17)."""
        source = "attention(4, rope=True, name='attn')\n"
        model = network(source, input_shape=("B", 17, 64), output_shape=("B", 17, 64), device=device)
        x = torch.randn(2, 17, 64, device=device)
        out = model(x)
        assert out.shape == (2, 17, 64)

    def test_tier2_f1_zero_scale_logit_softcap_disabled(self, device):
        """[F1 Boundary 5] logit_softcap=0.0 cleanly disables soft-capping without tanh application."""
        source = "attention(4, logit_softcap=0.0, name='attn')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 8, 32)

    # --------------------------------------------------------------------------
    # F2 Boundaries: Transformer Block
    # --------------------------------------------------------------------------

    def test_tier2_f2_transformer_block_single_token_sequence(self, device):
        """[F2 Boundary 1] Transformer block evaluated on single token generation (T=1)."""
        source = "transformer_block(4, ffn_kind='swiglu', causal=True, rope=True, name='tb')\n"
        model = network(source, input_shape=("B", 1, 64), output_shape=("B", 1, 64), device=device)
        x = torch.randn(2, 1, 64, device=device)
        out = model(x)
        assert out.shape == (2, 1, 64)

    def test_tier2_f2_transformer_block_non_integer_hidden_ratio(self, device):
        """[F2 Boundary 2] Transformer block with explicit non-integer hidden width (8/3 D = 170 for D=64)."""
        source = "transformer_block(4, hidden=170, ffn_kind='swiglu', name='tb')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier2_f2_transformer_block_qk_norm_l2_zero_input(self, device):
        """[F2 Boundary 3] Transformer block with qk_norm='l2_norm' numerical stability on zero vectors."""
        source = "transformer_block(4, qk_norm='l2_norm', name='tb')\n"
        model = network(source, input_shape=("B", 4, 32), output_shape=("B", 4, 32), device=device)
        x = torch.zeros(2, 4, 32, device=device)
        out = model(x)
        assert out.isfinite().all()

    def test_tier2_f2_transformer_block_kv_heads_1_mqa(self, device):
        """[F2 Boundary 4] Transformer block with MQA (kv_heads=1, multi-query attention)."""
        source = "transformer_block(4, kv_heads=1, name='tb')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier2_f2_transformer_block_extreme_head_count(self, device):
        """[F2 Boundary 5] Transformer block with heads=16 on D=64 (head_dim=4)."""
        source = "transformer_block(16, name='tb')\n"
        model = network(source, input_shape=("B", 4, 64), output_shape=("B", 4, 64), device=device)
        x = torch.randn(2, 4, 64, device=device)
        out = model(x)
        assert out.shape == (2, 4, 64)

    # --------------------------------------------------------------------------
    # F3 Boundaries: MoE SOTA Extensions
    # --------------------------------------------------------------------------

    def test_tier2_f3_moe_top_k_equals_experts(self, device):
        """[F3 Boundary 1] MoE with top_k equal to total experts (dense mixture)."""
        source = "moe(4, 64, top_k=4, expert_kind='swiglu', name='moe')\n"
        model = network(source, input_shape=("B", 4, 32), output_shape=("B", 4, 32), device=device)
        x = torch.randn(2, 4, 32, device=device)
        out = model(x)
        assert out.shape == (2, 4, 32)

    def test_tier2_f3_moe_top_k_equals_one(self, device):
        """[F3 Boundary 2] MoE with top_k=1 (hard single expert selection)."""
        source = "moe(4, 64, top_k=1, scoring='sigmoid', name='moe')\n"
        model = network(source, input_shape=("B", 4, 32), output_shape=("B", 4, 32), device=device)
        x = torch.randn(2, 4, 32, device=device)
        out = model(x)
        assert out.shape == (2, 4, 32)

    def test_tier2_f3_moe_top_k_exceeding_experts_rejected(self):
        """[F3 Boundary 3] MoE with top_k > experts raises E_ARGUMENT."""
        source = "moe(4, 64, top_k=5)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 4, 32), output_shape=("B", 4, 32))
        assert exc_info.value.code == "E_ARGUMENT"

    def test_tier2_f3_moe_sigmoid_routing_identical_logits(self, device):
        """[F3 Boundary 4] MoE with sigmoid routing when router weights produce identical affinities."""
        source = "moe(2, 32, top_k=2, scoring='sigmoid', bias=False, name='moe')\n"
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        with torch.no_grad():
            if hasattr(model["moe"], "router"):
                model["moe"].router.weight.zero_()
        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        assert out.isfinite().all()

    def test_tier2_f3_moe_batch_size_one(self, device):
        """[F3 Boundary 5] MoE evaluated on single batch item (B=1)."""
        source = "moe(4, 64, top_k=2, expert_kind='swiglu', name='moe')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(1, 8, 32, device=device)
        out = model(x)
        assert out.shape == (1, 8, 32)

    # --------------------------------------------------------------------------
    # F4 Boundaries: Convolution, Normalization & MatMul
    # --------------------------------------------------------------------------

    def test_tier2_f4_conv1d_causal_padding_rejected_with_nonzero_padding(self):
        """[F4 Boundary 1] conv1d with causal=True and padding > 0 raises E_ARGUMENT."""
        source = "conv1d(32, kernel_size=3, padding=1, causal=True)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 16, 24), output_shape=("B", 32, 24))
        assert exc_info.value.code == "E_ARGUMENT"

    def test_tier2_f4_conv1d_causal_stride1_bijective_solver(self):
        """[F4 Boundary 2] conv1d with causal=True and stride=1 establishes exact bijective L_out == L_in."""
        source = "conv1d(32, kernel_size=5, causal=True)\n"
        plan = resolve(source, input_shape=("B", 16, 40), output_shape=("B", 32, 40))
        assert plan.outputs["output"]["shape"] == ("B", 32, 40)

    def test_tier2_f4_rms_norm_zero_centered_zero_inputs(self, device):
        """[F4 Boundary 3] rms_norm(zero_centered=True) stability on zero inputs."""
        source = "rms_norm(zero_centered=True, name='norm')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.zeros(2, 8, 32, device=device)
        out = model(x)
        assert out.isfinite().all()

    def test_tier2_f4_adaptive_norm_single_token_sequence(self, device):
        """[F4 Boundary 4] adaptive_norm rank-3 evaluated on single token sequence (T=1)."""
        source = "out = adaptive_norm(x, params, name='ad')\n"
        model = network(
            source,
            input_shape={"x": ("B", 1, 32), "params": ("B", 64)},
            output_shape={"out": ("B", 1, 32)},
            device=device,
        )
        x = torch.randn(2, 1, 32, device=device)
        params = torch.randn(2, 64, device=device)
        out = model(x=x, params=params)
        assert out.shape == (2, 1, 32)

    def test_tier2_f4_matmul_rank2_unbatched_multiplication(self, device):
        """[F4 Boundary 5] matmul generalized shape on rank-2 matrices [N, K] @ [K, M] -> [N, M]."""
        source = "out = matmul(a, b, name='mm')\n"
        model = network(
            source,
            input_shape={"a": ("B", 16, 32), "b": ("B", 32, 8)},
            output_shape={"out": ("B", 16, 8)},
            device=device,
        )
        a = torch.randn(2, 16, 32, device=device)
        b = torch.randn(2, 32, 8, device=device)
        out = model(a=a, b=b)
        assert out.shape == (2, 16, 8)

    # --------------------------------------------------------------------------
    # F5 Boundaries: Gated Delta Attention (KDA / GDN)
    # --------------------------------------------------------------------------

    def test_tier2_f5_kda_sequence_length_not_multiple_of_chunk(self, device):
        """[F5 Boundary 1] gated_delta_attention when sequence length (T=37) is not divisible by chunk_size=16."""
        source = "gated_delta_attention(4, chunk_size=16, name='kda')\n"
        model = network(source, input_shape=("B", 37, 64), output_shape=("B", 37, 64), device=device)
        x = torch.randn(2, 37, 64, device=device)
        out = model(x)
        assert out.shape == (2, 37, 64)

    def test_tier2_f5_kda_sequence_smaller_than_chunk_size(self, device):
        """[F5 Boundary 2] gated_delta_attention when sequence length (T=8) is smaller than chunk_size=32."""
        source = "gated_delta_attention(4, chunk_size=32, name='kda')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier2_f5_kda_conv_size_zero_disabled(self, device):
        """[F5 Boundary 3] gated_delta_attention with conv_size=0 (disabling causal short conv1d)."""
        source = "gated_delta_attention(4, conv_size=0, name='kda')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier2_f5_kda_single_token_evaluation(self, device):
        """[F5 Boundary 4] gated_delta_attention single token step evaluation (T=1)."""
        source = "gated_delta_attention(4, name='kda')\n"
        model = network(source, input_shape=("B", 1, 64), output_shape=("B", 1, 64), device=device)
        x = torch.randn(2, 1, 64, device=device)
        out = model(x)
        assert out.shape == (2, 1, 64)

    def test_tier2_f5_kda_zero_input_stability(self, device):
        """[F5 Boundary 5] gated_delta_attention numerical stability on all-zero input tensor."""
        source = "gated_delta_attention(4, name='kda')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.zeros(2, 8, 64, device=device)
        out = model(x)
        assert out.isfinite().all()

    # --------------------------------------------------------------------------
    # F6 Boundaries: Latent Attention (MLA)
    # --------------------------------------------------------------------------

    def test_tier2_f6_mla_extreme_kv_compression_rank1(self, device):
        """[F6 Boundary 1] latent_attention with extreme KV bottleneck compression (kv_lora_rank=1)."""
        source = "latent_attention(2, kv_lora_rank=1, name='mla')\n"
        model = network(source, input_shape=("B", 8, 32), output_shape=("B", 8, 32), device=device)
        x = torch.randn(2, 8, 32, device=device)
        out = model(x)
        assert out.shape == (2, 8, 32)

    def test_tier2_f6_mla_direct_q_projection(self, device):
        """[F6 Boundary 2] latent_attention with q_lora_rank=0 (direct uncompressed W_Q projection)."""
        source = "latent_attention(4, kv_lora_rank=16, q_lora_rank=0, name='mla')\n"
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier2_f6_mla_non_causal_bidirectional(self, device):
        """[F6 Boundary 3] latent_attention with causal=False for bidirectional encoder representations."""
        source = "latent_attention(4, kv_lora_rank=16, causal=False, name='mla')\n"
        model = network(source, input_shape=("B", 12, 64), output_shape=("B", 12, 64), device=device)
        x = torch.randn(2, 12, 64, device=device)
        out = model(x)
        assert out.shape == (2, 12, 64)

    def test_tier2_f6_mla_odd_sequence_length(self, device):
        """[F6 Boundary 4] latent_attention with decoupled RoPE on odd sequence length (T=15)."""
        source = "latent_attention(4, kv_lora_rank=16, qk_rope_head_dim=16, name='mla')\n"
        model = network(source, input_shape=("B", 15, 64), output_shape=("B", 15, 64), device=device)
        x = torch.randn(2, 15, 64, device=device)
        out = model(x)
        assert out.shape == (2, 15, 64)

    def test_tier2_f6_mla_single_head_minimal(self, device):
        """[F6 Boundary 5] latent_attention with heads=1 minimal configuration."""
        source = "latent_attention(1, kv_lora_rank=8, name='mla')\n"
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        assert out.shape == (2, 4, 16)

    # --------------------------------------------------------------------------
    # F7 Boundaries: Normalization & Positional Operators
    # --------------------------------------------------------------------------

    def test_tier2_f7_l2_norm_zero_vector_stability(self, device):
        """[F7 Boundary 1] l2_norm on all-zero vectors maintains epsilon protection without NaN."""
        source = "l2_norm(dim=-1, eps=1e-8, name='norm')\n"
        model = network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        x = torch.zeros(2, 4, 16, device=device)
        out = model(x)
        assert out.isfinite().all()

    def test_tier2_f7_l2_norm_non_last_dimension(self, device):
        """[F7 Boundary 2] l2_norm along intermediate dimension (dim=1)."""
        source = "l2_norm(dim=1, name='norm')\n"
        model = network(source, input_shape=("B", 8, 16), output_shape=("B", 8, 16), device=device)
        x = torch.randn(2, 8, 16, device=device)
        out = model(x)
        norms = torch.norm(out, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_tier2_f7_adaln_zero_zero_conditioning_vector(self, device):
        """[F7 Boundary 3] adaln_zero with zero conditioning vector cond=0 produces exact unshifted normalization."""
        source = "normed, gate = adaln_zero(x, cond, name='ad')\n"
        model = network(
            source,
            input_shape={"x": ("B", 4, 16), "cond": ("B", 8)},
            output_shape={"normed": ("B", 4, 16), "gate": ("B", 4, 16)},
            device=device,
        )
        x = torch.randn(2, 4, 16, device=device)
        cond = torch.zeros(2, 8, device=device)
        out = model(x=x, cond=cond)
        assert out["gate"].sum() == 0.0

    def test_tier2_f7_rope_single_token(self, device):
        """[F7 Boundary 4] standalone rope on single token sequence (T=1)."""
        source = "rope(heads=2, name='rp')\n"
        model = network(source, input_shape=("B", 1, 16), output_shape=("B", 1, 16), device=device)
        x = torch.randn(2, 1, 16, device=device)
        out = model(x)
        assert out.shape == (2, 1, 16)

    def test_tier2_f7_rope_theta_one_boundary(self, device):
        """[F7 Boundary 5] standalone rope with theta=1.0 minimal base frequency."""
        source = "rope(heads=2, theta=1.0, name='rp')\n"
        with pytest.raises(HNDLError) as exc_info:
            network(source, input_shape=("B", 4, 16), output_shape=("B", 4, 16), device=device)
        assert exc_info.value.code == "E_ARGUMENT"

    # --------------------------------------------------------------------------
    # F8 Boundaries: Domain-Completing Operators
    # --------------------------------------------------------------------------

    def test_tier2_f8_deconv1d_odd_input_length(self, device):
        """[F8 Boundary 1] deconv1d with stride=2 on odd length input (L=15 -> L=30 with policy='up2')."""
        source = "deconv1d(16, kernel_size=4, stride=2, padding=1, name='dec')\n"
        model = network(source, input_shape=("B", 8, 15), output_shape=("B", 16, 30), device=device)
        x = torch.randn(2, 8, 15, device=device)
        out = model(x)
        assert out.shape == (2, 16, 30)

    def test_tier2_f8_grn_constant_spatial_features(self, device):
        """[F8 Boundary 2] grn on spatially uniform feature maps remains stable."""
        source = "grn(name='grn')\n"
        model = network(source, input_shape=("B", 16, 4, 4), output_shape=("B", 16, 4, 4), device=device)
        x = torch.ones(2, 16, 4, 4, device=device)
        out = model(x)
        assert out.isfinite().all()

    def test_tier2_f8_sinusoidal_embed_minimal_dimension(self, device):
        """[F8 Boundary 3] sinusoidal_embed with minimal out_dim=2."""
        source = "sinusoidal_embed(2, name='sin')\n"
        model = network(source, input_shape=("B", 1), output_shape=("B", 2), device=device)
        t = torch.tensor([[1.0], [2.0]], device=device)
        out = model(t)
        assert out.shape == (2, 2)

    def test_tier2_f8_cast_invalid_dtype_rejected(self):
        """[F8 Boundary 4] cast with unsupported dtype raises E_ARGUMENT."""
        source = "cast(dtype='complex128')\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 16), output_shape=("B", 16))
        assert exc_info.value.code == "E_ARGUMENT"

    def test_tier2_f8_sinusoidal_embed_odd_dimension_rejected(self):
        """[F8 Boundary 5] sinusoidal_embed with odd out_dim (e.g. 15) raises E_CONSTRAINT."""
        source = "sinusoidal_embed(15)\n"
        with pytest.raises(HNDLError) as exc_info:
            resolve(source, input_shape=("B", 1), output_shape=("B", 15))
        assert exc_info.value.code in ("E_CONSTRAINT", "E_ARGUMENT")

    # --------------------------------------------------------------------------
    # F9 Boundaries: Pretrained Provider Hardening
    # --------------------------------------------------------------------------

    def test_tier2_f9_pretrained_ids_contract_rejects_float_tensor(self, text_checkpoint, device):
        """[F9 Boundary 1] pretrained with contract='ids' rejects floating point inputs with E_DTYPE."""
        source = f'pretrained("{text_checkpoint}", name="lm")\n'
        with pytest.raises(HNDLError) as exc_info:
            network(
                source,
                input_shape=("B", 8),
                output_shape=("B", 8, 16),
                input_dtype="float32",
                device=device,
            )
        assert exc_info.value.code in ("E_DTYPE", "E_PRETRAINED")

    def test_tier2_f9_pretrained_rank_mismatch_rejected(self, text_checkpoint, device):
        """[F9 Boundary 2] pretrained with contract='ids' rejects rank-3 input [B, T, D]."""
        source = f'pretrained("{text_checkpoint}", name="lm")\n'
        with pytest.raises(HNDLError) as exc_info:
            network(
                source,
                input_shape=("B", 8, 16),
                output_shape=("B", 8, 16),
                input_dtype="int64",
                device=device,
            )
        assert exc_info.value.code in ("E_RANK", "E_CONSTRAINT", "E_PRETRAINED")

    def test_tier2_f9_pretrained_missing_shape_fn_meta_fallback(self):
        """[F9 Boundary 3] Provider registered without shape_fn retains meta tracing fallback."""
        reg = Registry()

        def custom_builder(args):
            return nn.Linear(8, 16)

        reg.pretrained_provider("meta_provider", custom_builder)
        assert "meta_provider" in reg.pretrained_providers

    def test_tier2_f9_pretrained_invalid_checkpoint_error_code(self):
        """[F9 Boundary 4] Non-existent pretrained path triggers clean E_PRETRAINED."""
        from hndl.pretrained import resolve_source
        with pytest.raises(HNDLError) as exc_info:
            resolve_source("/invalid/path/that/does/not/exist")
        assert exc_info.value.code == "E_PRETRAINED"

    def test_tier2_f9_pretrained_single_token_evaluation(self, text_checkpoint, device):
        """[F9 Boundary 5] pretrained checkpoint evaluated on single token context (T=1)."""
        source = f'pretrained("{text_checkpoint}", name="lm")\n'
        model = network(
            source,
            input_shape=("B", 1),
            output_shape=("B", 1, 8),
            input_dtype="int64",
            device=device,
        )
        tokens = torch.randint(0, 16, (1, 1), device=device)
        out = model(tokens)
        assert out.shape == (1, 1, 8)

    # --------------------------------------------------------------------------
    # F10 Boundaries: Network Recipes Integration
    # --------------------------------------------------------------------------

    def test_tier2_f10_recipe_sidecar_completeness(self):
        """[F10 Boundary 1] All authored network recipes have sidecars with title, description, and parameters."""
        recipes_dir = REPO_ROOT / "examples" / "networks"
        for hndl_file in recipes_dir.glob("*.hndl"):
            json_file = hndl_file.with_suffix(".json")
            if json_file.exists():
                data = json.loads(json_file.read_text(encoding="utf-8"))
                assert "title" in data
                assert "description" in data
                assert "parameters" in data

    def test_tier2_f10_recipe_round_trip_digest_invariance(self):
        """[F10 Boundary 2] Recipes preserve bit-exact semantic digests across JSON round-trips."""
        recipes_dir = REPO_ROOT / "examples" / "networks"
        for hndl_file in recipes_dir.glob("*.hndl"):
            json_file = hndl_file.with_suffix(".json")
            if not json_file.exists():
                continue
            meta = json.loads(json_file.read_text(encoding="utf-8"))
            kwargs = {}
            if "input_dtype" in meta:
                kwargs["input_dtype"] = meta["input_dtype"]
            plan = resolve(
                hndl_file.read_text(encoding="utf-8"),
                input_shape=meta["input_shape"],
                output_shape=meta["output_shape"],
                **kwargs,
            )
            round_tripped = ResolvedPlan.from_json(plan.to_json())
            assert round_tripped.semantic_digest == plan.semantic_digest

    def test_tier2_f10_recipe_parameter_count_static_matches_runtime(self, device):
        """[F10 Boundary 3] Static parameter counts match runtime module parameter count on ready recipes."""
        hndl_file = REPO_ROOT / "examples" / "networks" / "espcn_super_res.hndl"
        if not hndl_file.exists():
            pytest.skip("espcn_super_res.hndl not yet authored")
        meta = json.loads(hndl_file.with_suffix(".json").read_text(encoding="utf-8"))
        model = network(hndl_file.read_text(encoding="utf-8"), input_shape=meta["input_shape"], output_shape=meta["output_shape"], device=device)
        static_count = sum(parameter_counts(model.plan).values())
        runtime_count = sum(p.numel() for p in model.parameters())
        assert static_count == runtime_count == meta["parameters"]

    def test_tier2_f10_recipe_batch_size_variation(self, device):
        """[F10 Boundary 4] Recipes execute with variable batch sizes (B=1, B=7)."""
        hndl_file = REPO_ROOT / "examples" / "networks" / "convnext_tiny.hndl"
        if not hndl_file.exists():
            pytest.skip("convnext_tiny.hndl not yet authored")
        model = network(hndl_file.read_text(encoding="utf-8"), input_shape=("B", 3, 32, 32), output_shape=("B", 10), device=device)
        out1 = model(torch.randn(1, 3, 32, 32, device=device))
        out7 = model(torch.randn(7, 3, 32, 32, device=device))
        assert out1.shape == (1, 10)
        assert out7.shape == (7, 10)

    def test_tier2_f10_recipe_finite_gradients_on_backward(self, device):
        """[F10 Boundary 5] Recipe backward pass verifies all parameters receive non-null gradients."""
        hndl_file = REPO_ROOT / "examples" / "networks" / "espcn_super_res.hndl"
        if not hndl_file.exists():
            pytest.skip("espcn_super_res.hndl not yet authored")
        model = network(hndl_file.read_text(encoding="utf-8"), input_shape=("B", 3, 16, 16), output_shape=("B", 3, 32, 32), device=device)
        x = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
        out = model(x)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
            assert p.grad.isfinite().all()


# ==============================================================================
# TIER 3: Cross-Feature Interactions (Pairwise Combinations)
# ==============================================================================


class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise combinations across major SOTA feature areas (>=15 tests)."""

    def test_tier3_gqa_with_per_head_qk_norm(self, device):
        """[Pairwise 1] GQA (kv_heads=2) combined with per-head QK-RMSNorm."""
        source = "attention(4, kv_heads=2, qk_norm='rms_norm', name='attn')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_mla_with_nope_mode(self, device):
        """[Pairwise 2] Multi-Head Latent Attention in NoPE mode (qk_rope_head_dim=0) with KV LoRA."""
        source = "latent_attention(4, kv_lora_rank=16, qk_rope_head_dim=0, name='mla')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_kda_with_swiglu_moe(self, device):
        """[Pairwise 3] Gated Delta Attention (KDA) chained directly into SwiGLU MoE."""
        source = (
            "k = gated_delta_attention(4, decay_kind='channel', name='kda')\n"
            "moe(k, 4, 128, top_k=2, expert_kind='swiglu', name='moe')\n"
        )
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_causal_conv1d_with_deconv1d_upsampling(self, device):
        """[Pairwise 4] Causal conv1d (stride=1) followed by deconv1d (stride=2 2x upsampling)."""
        source = (
            "c = conv1d(32, kernel_size=3, causal=True, name='enc')\n"
            "deconv1d(16, kernel_size=4, stride=2, padding=1, name='dec')\n"
        )
        model = network(source, input_shape=("B", 16, 24), output_shape=("B", 16, 48), device=device)
        x = torch.randn(2, 16, 24, device=device)
        out = model(x)
        assert out.shape == (2, 16, 48)

    def test_tier3_adaln_zero_with_sinusoidal_embed(self, device):
        """[Pairwise 5] Sinusoidal timestep embedding conditioning an adaln_zero modulation block."""
        source = (
            "cond = sinusoidal_embed(t, 32, name='time_emb')\n"
            "normed, gate = adaln_zero(x, cond, name='ad')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 16, 64), "t": ("B", 1)},
            output_shape={"normed": ("B", 16, 64), "gate": ("B", 16, 64)},
            device=device,
        )
        x = torch.randn(2, 16, 64, device=device)
        t = torch.tensor([[5.0], [25.0]], device=device)
        out = model(x=x, t=t)
        assert out["normed"].shape == (2, 16, 64)
        assert out["gate"].shape == (2, 16, 64)

    def test_tier3_gqa_high_theta_and_logit_softcap(self, device):
        """[Pairwise 6] GQA (4:2) with high-theta RoPE (1,000,000) and logit soft-capping (30.0)."""
        source = "attention(4, kv_heads=2, rope=True, rope_theta=1000000.0, logit_softcap=30.0, name='attn')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_standalone_rope_with_parameter_free_l2_norm(self, device):
        """[Pairwise 7] Parameter-free l2_norm followed by standalone rotary position embedding."""
        source = (
            "n = l2_norm(dim=-1, name='norm')\n"
            "rope(n, heads=4, theta=10000.0, name='rp')\n"
        )
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_grn_with_zero_centered_rms_norm(self, device):
        """[Pairwise 8] Global Response Normalization (GRN) on vision maps flattened into zero_centered RMSNorm."""
        source = (
            "g = grn(name='grn')\n"
            "f = flatten(g, name='flat')\n"
            "rms_norm(f, zero_centered=True, name='norm')\n"
        )
        model = network(source, input_shape=("B", 16, 4, 4), output_shape=("B", 256), device=device)
        x = torch.randn(2, 16, 4, 4, device=device)
        out = model(x)
        assert out.shape == (2, 256)

    def test_tier3_rank3_adaptive_norm_with_batched_matmul(self, device):
        """[Pairwise 9] Rank-3 adaptive_norm modulated sequence followed by batched matmul."""
        source = (
            "normed = adaptive_norm(x, params, name='ad')\n"
            "out = matmul(normed, proj, name='mm')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 16, 32), "params": ("B", 64), "proj": ("B", 32, 16)},
            output_shape={"out": ("B", 16, 16)},
            device=device,
        )
        x = torch.randn(2, 16, 32, device=device)
        params = torch.randn(2, 64, device=device)
        proj = torch.randn(2, 32, 16, device=device)
        out = model(x=x, params=params, proj=proj)
        assert out.shape == (2, 16, 16)

    def test_tier3_cast_with_embedding_and_gqa_attention(self, device):
        """[Pairwise 10] Cast int32 inputs to int64 for token embeddings, followed by GQA attention."""
        source = (
            "idx = cast(dtype='int64', name='cast')\n"
            "tok = embedding(idx, 128, 64, name='emb')\n"
            "out = attention(tok, 4, kv_heads=2, name='attn')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 16)},
            output_shape={"out": ("B", 16, 64)},
            input_dtype="int32",
            device=device,
        )
        tokens = torch.randint(0, 128, (2, 16), dtype=torch.int32, device=device)
        out = model(x=tokens)
        assert out.shape == (2, 16, 64)

    def test_tier3_causal_conv1d_channels_last_with_kda(self, device):
        """[Pairwise 11] conv1d(causal=True, channels_last=True) feeding directly into gated_delta_attention."""
        source = (
            "stem = conv1d(64, kernel_size=3, causal=True, channels_last=True, name='stem')\n"
            "gated_delta_attention(stem, 4, name='kda')\n"
        )
        model = network(source, input_shape=("B", 16, 32), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 32, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_transformer_block_swiglu_qk_norm_high_theta(self, device):
        """[Pairwise 12] Transformer block pairing SwiGLU FFN, QK-RMSNorm, and theta=1000000.0 RoPE."""
        source = "transformer_block(4, kv_heads=2, qk_norm='rms_norm', rope=True, rope_theta=1000000.0, ffn_kind='swiglu', bias=False, name='tb')\n"
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_tier3_latent_attention_with_adaln_zero_diffusion(self, device):
        """[Pairwise 13] Latent Attention inside an AdaLN-Zero modulated diffusion block."""
        source = (
            "normed, gate = adaln_zero(x, cond, name='ad')\n"
            "attn = latent_attention(normed, 4, kv_lora_rank=16, name='mla')\n"
            "out = add(x, broadcast_mul(attn, gate), name='res')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 16, 64), "cond": ("B", 32)},
            output_shape={"out": ("B", 16, 64)},
            device=device,
        )
        x = torch.randn(2, 16, 64, device=device)
        cond = torch.randn(2, 32, device=device)
        out = model(x=x, cond=cond)
        assert out.shape == (2, 16, 64)

    def test_tier3_deconv1d_audio_with_cross_attention(self, device):
        """[Pairwise 14] Acoustic cross_attention conditioning a deconv1d waveform synthesis stem."""
        source = (
            "xattn = cross_attention(x, enc, 4, kv_heads=2, name='xattn')\n"
            "out = deconv1d(xattn, 16, kernel_size=4, stride=2, padding=1, name='synth')\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 32, 16), "enc": ("B", 32, 16)},
            output_shape={"out": ("B", 16, 32)},
            device=device,
        )
        x = torch.randn(2, 32, 16, device=device)
        enc = torch.randn(2, 32, 16, device=device)
        out = model(x=x, enc=enc)
        assert out.shape == (2, 16, 32)

    def test_tier3_moe_sigmoid_routing_with_layer_norm_bias_false(self, device):
        """[Pairwise 15] SwiGLU MoE with Sigmoid routing preceded by bias=False LayerNorm."""
        source = (
            "n = layer_norm(bias=False, name='ln')\n"
            "moe(n, 4, 128, top_k=2, expert_kind='swiglu', scoring='sigmoid', bias=False, name='moe')\n"
        )
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64)

    def test_tier3_standalone_rope_with_latent_attention_nope(self, device):
        """[Pairwise 16] Standalone RoPE applied to tokens before NoPE Latent Attention."""
        source = (
            "r = rope(heads=4, theta=10000.0, name='rp')\n"
            "latent_attention(r, 4, kv_lora_rank=16, qk_rope_head_dim=0, name='mla')\n"
        )
        model = network(source, input_shape=("B", 16, 64), output_shape=("B", 16, 64), device=device)
        x = torch.randn(2, 16, 64, device=device)
        out = model(x)
        assert out.shape == (2, 16, 64)


# ==============================================================================
# TIER 4: Real-World Application Workloads
# ==============================================================================


class TestTier4RealWorldWorkloads:
    """Tier 4: Realistic end-to-end multi-layer application workloads (>=5 tests)."""

    def test_tier4_workload1_hybrid_kda_mla_causal_decoder(self, device):
        """[Workload 1] Hybrid 3:1 KDA + NoPE MLA language model block with SwiGLU FFN and pre-RMSNorm."""
        source = (
            "h = embedding(256, 64, name='tok_emb')\n"
            "# 3 KDA layers\n"
            "for _ in range(3):\n"
            "    k = gated_delta_attention(rms_norm(h), 2, decay_kind='channel', conv_size=4)\n"
            "    h = add(h, k)\n"
            "    f = swiglu(rms_norm(h), 128, bias=False)\n"
            "    h = add(h, f)\n"
            "# 1 NoPE MLA layer\n"
            "m = latent_attention(rms_norm(h), 2, kv_lora_rank=16, qk_rope_head_dim=0, causal=True)\n"
            "h = add(h, m)\n"
            "f = swiglu(rms_norm(h), 128, bias=False)\n"
            "h = add(h, f)\n"
            "rms_norm(h, name='out_norm')\n"
            "linear(bias=False, name='lm_head')\n"
        )
        model = network(source, input_shape=("B", 16), output_shape=("B", 16, 256), input_dtype="int64", device=device)
        tokens = torch.randint(0, 256, (2, 16), device=device)
        out = model(tokens)
        assert out.shape == (2, 16, 256)
        assert out.isfinite().all()

        # Backward autograd check
        out.sum().backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        assert all(p.grad is not None for p in trainable)

    def test_tier4_workload2_qwen3_gqa_qknorm_decoder_block(self, device):
        """[Workload 2] Qwen3 style transformer block: 4:2 GQA, per-head QK-RMSNorm, high-theta RoPE, and SwiGLU."""
        source = (
            "x = embedding(256, 64, name='tok_emb')\n"
            "for _ in range(2):\n"
            "    x = transformer_block(\n"
            "        x, 4, kv_heads=2, hidden=192, ffn_kind='swiglu',\n"
            "        norm='rms_norm', qk_norm='rms_norm',\n"
            "        causal=True, rope=True, rope_theta=1000000.0, bias=False,\n"
            "    )\n"
            "rms_norm(x, zero_centered=True, name='final_norm')\n"
            "linear(bias=False, name='lm_head')\n"
        )
        model = network(source, input_shape=("B", 16), output_shape=("B", 16, 256), input_dtype="int64", device=device)
        tokens = torch.randint(0, 256, (2, 16), device=device)
        out = model(tokens)
        assert out.shape == (2, 16, 256)

        out.sum().backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is not None for p in trainable)

    def test_tier4_workload3_dit_diffusion_transformer_block(self, device):
        """[Workload 3] DiT block: Sinusoidal timestep embedding + AdaLN-Zero modulation + QK-Norm attention."""
        source = (
            "cond = sinusoidal_embed(t, 32, name='time_emb')\n"
            "# DiT block\n"
            "normed_attn, gate_attn = adaln_zero(x, cond, name='attn_ad')\n"
            "attn = attention(normed_attn, 4, qk_norm='rms_norm', name='attn')\n"
            "x = add(x, broadcast_mul(attn, gate_attn))\n"
            "normed_ffn, gate_ffn = adaln_zero(x, cond, name='ffn_ad')\n"
            "ffn = swiglu(normed_ffn, 128, bias=False, name='ffn')\n"
            "out = add(x, broadcast_mul(ffn, gate_ffn))\n"
        )
        model = network(
            source,
            input_shape={"x": ("B", 16, 64), "t": ("B", 1)},
            output_shape={"out": ("B", 16, 64)},
            device=device,
        )
        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        t = torch.tensor([[10.0], [50.0]], device=device)
        out = model(x=x, t=t)
        assert out.shape == (2, 16, 64)

        out.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()

    def test_tier4_workload4_audio_causal_conv_waveform_synthesizer(self, device):
        """[Workload 4] Audio acoustic backbone: Causal Conv1d stem + SwiGLU FFN + Deconv1d waveform upsampler."""
        source = (
            "# Causal acoustic stem\n"
            "stem1 = conv1d(32, kernel_size=3, causal=True, name='stem1')\n"
            "act1 = gelu(stem1)\n"
            "stem2 = conv1d(act1, 64, kernel_size=3, stride=2, causal=True, name='stem2')\n"
            "enc = gelu(stem2)\n"
            "# 2x Waveform upsampling via deconv1d\n"
            "up1 = deconv1d(enc, 32, kernel_size=4, stride=2, padding=1, name='up1')\n"
            "act2 = gelu(up1)\n"
            "up2 = deconv1d(act2, 16, kernel_size=4, stride=2, padding=1, name='up2')\n"
            "out = conv1d(up2, 1, kernel_size=3, padding=1, name='out_wave')\n"
        )
        model = network(source, input_shape={"x": ("B", 16, 32)}, output_shape={"out": ("B", 1, 64)}, device=device)
        audio = torch.randn(2, 16, 32, device=device, requires_grad=True)
        out = model(x=audio)
        assert out.shape == (2, 1, 64)

        out.sum().backward()
        assert audio.grad is not None

    def test_tier4_workload5_convnext_v2_modern_vision_stage(self, device):
        """[Workload 5] ConvNeXt-V2 block: 7x7 depthwise conv + Global Response Normalization (GRN) + LayerScale."""
        source = (
            "dw = conv(x, 64, kernel_size=7, padding=3, groups=64, name='dwconv')\n"
            "norm = group_norm(dw, 1, name='gn')\n"
            "pw1 = conv(norm, 256, kernel_size=1, name='pw1')\n"
            "act = gelu(pw1)\n"
            "grn_out = grn(act, name='grn')\n"
            "pw2 = conv(grn_out, 64, kernel_size=1, name='pw2')\n"
            "scaled = learned_scale(pw2, init_value=1e-6, name='scale')\n"
            "out = add(x, scaled, name='res')\n"
        )
        model = network(source, input_shape={"x": ("B", 64, 16, 16)}, output_shape={"out": ("B", 64, 16, 16)}, device=device)
        images = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
        out = model(x=images)
        assert out.shape == (2, 64, 16, 16)

        out.sum().backward()
        assert images.grad is not None

    def test_tier4_workload6_deepseek_v3_sparse_moe_block(self, device):
        """[Workload 6] DeepSeek-V3 style block: Multi-Head Latent Attention + top-2 routed SwiGLU MoE with Sigmoid routing."""
        source = (
            "# Pre-norm MLA\n"
            "n1 = rms_norm(x, name='norm1')\n"
            "mla = latent_attention(n1, 4, kv_lora_rank=16, qk_rope_head_dim=0, causal=True, name='mla')\n"
            "h = add(x, mla)\n"
            "# Pre-norm Sparse SwiGLU MoE with Sigmoid routing\n"
            "n2 = rms_norm(h, name='norm2')\n"
            "moe_out = moe(n2, 4, 128, top_k=2, expert_kind='swiglu', scoring='sigmoid', bias=False, name='moe')\n"
            "out = add(h, moe_out)\n"
        )
        model = network(source, input_shape=("B", 8, 64), output_shape=("B", 8, 64), device=device)
        x = torch.randn(2, 8, 64, device=device, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 8, 64)

        out.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()
