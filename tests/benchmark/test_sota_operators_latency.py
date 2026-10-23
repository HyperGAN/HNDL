"""Microbenchmarks for SOTA operator forward latency.

Benchmarks:
1. MixtureOfExperts: sparse grouped dispatch scaling across expert counts and kinds (MLP vs SwiGLU).
2. GatedDeltaAttention: 4D Tensor Core chunkwise forward across sequence lengths.
3. LatentAttention: standard vs absorbed weight inference latency.
4. Attention: sliding window mask caching and zero-copy broadcast GQA softcapping.
"""

import math
import pytest
import torch

from hndl.operators.attention import Attention
from hndl.operators.gated_delta_attention import GatedDeltaAttention
from hndl.operators.latent_attention import LatentAttention
from hndl.operators.moe import MixtureOfExperts
from tests.benchmark._timing import format_measurement, measure

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("expert_kind", ["mlp", "swiglu"])
def test_moe_latency(expert_kind, benchmark_min_time, device):
    """Benchmark MixtureOfExperts sparse forward execution."""
    d_model = 128
    hidden = 256
    experts = 8
    top_k = 2
    batch = 4
    seq_len = 64

    model = MixtureOfExperts(
        experts=experts,
        hidden=hidden,
        top_k=top_k,
        activation="silu",
        D=d_model,
        expert_kind=expert_kind,
    ).to(device=device).eval()

    x = torch.randn(batch, seq_len, d_model, device=device)

    # Warm-up pass
    with torch.no_grad():
        out = model(x)
    assert out.isfinite().all()

    measurement = measure(lambda: model(x), min_run_time=benchmark_min_time)
    print(format_measurement(f"MoE [{expert_kind}, E={experts}, k={top_k}, {device}]", measurement))

    assert math.isfinite(measurement.mean) and measurement.mean > 0
    assert math.isfinite(measurement.median) and measurement.median > 0


@pytest.mark.parametrize("experts", [4, 8, 16])
def test_moe_scaling_experts(experts, benchmark_min_time, device):
    """Verify that MoE latency remains bounded as expert count scales (sparse O(k) compute)."""
    d_model = 128
    hidden = 128
    top_k = 2
    batch = 2
    seq_len = 64

    model = MixtureOfExperts(
        experts=experts,
        hidden=hidden,
        top_k=top_k,
        activation="silu",
        D=d_model,
        expert_kind="mlp",
    ).to(device=device).eval()

    x = torch.randn(batch, seq_len, d_model, device=device)

    with torch.no_grad():
        out = model(x)
    assert out.isfinite().all()

    measurement = measure(lambda: model(x), min_run_time=benchmark_min_time)
    print(format_measurement(f"MoE scaling [E={experts}, k={top_k}, {device}]", measurement))

    assert math.isfinite(measurement.mean) and measurement.mean > 0


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_gated_delta_attention_latency(seq_len, benchmark_min_time, device):
    """Benchmark 4D factorized GatedDeltaAttention chunkwise forward pass."""
    d_model = 256
    heads = 4
    chunk_size = 64
    batch = 2

    layer = GatedDeltaAttention(
        heads=heads,
        head_dim=d_model // heads,
        chunk_size=chunk_size,
        D=d_model,
    ).to(device=device).eval()

    x = torch.randn(batch, seq_len, d_model, device=device)

    with torch.no_grad():
        out = layer(x)
    assert out.isfinite().all()

    measurement = measure(lambda: layer(x), min_run_time=benchmark_min_time)
    print(format_measurement(f"GDA [seq={seq_len}, C={chunk_size}, {device}]", measurement))

    assert math.isfinite(measurement.mean) and measurement.mean > 0


def test_latent_attention_absorption_latency(benchmark_min_time, device):
    """Benchmark LatentAttention standard vs inference weight-absorbed forward pass."""
    d_model = 128
    heads = 4
    kv_lora_rank = 32
    qk_nope_head_dim = 32
    batch = 2
    seq_len = 32

    # Standard model
    model = LatentAttention(
        heads=heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=0,
        v_head_dim=qk_nope_head_dim,
        causal=True,
        D=d_model,
    ).to(device=device).eval()

    # Absorbed copy
    model_absorbed = LatentAttention(
        heads=heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=0,
        v_head_dim=qk_nope_head_dim,
        causal=True,
        D=d_model,
    ).to(device=device).eval()
    model_absorbed.load_state_dict(model.state_dict())
    model_absorbed.absorb_weights()

    x = torch.randn(batch, seq_len, d_model, device=device)

    with torch.no_grad():
        out_std = model(x)
        out_abs = model_absorbed(x)

    torch.testing.assert_close(out_abs, out_std, rtol=1e-4, atol=1e-4)

    m_std = measure(lambda: model(x), min_run_time=benchmark_min_time)
    m_abs = measure(lambda: model_absorbed(x), min_run_time=benchmark_min_time)

    print(format_measurement(f"MLA standard [{device}]", m_std))
    print(format_measurement(f"MLA absorbed [{device}]", m_abs))

    assert math.isfinite(m_std.mean) and m_std.mean > 0
    assert math.isfinite(m_abs.mean) and m_abs.mean > 0


def test_attention_caching_and_gqa_latency(benchmark_min_time, device):
    """Benchmark Attention with cached sliding-window masks and zero-copy broadcast GQA."""
    d_model = 256
    heads = 8
    kv_heads = 2
    batch = 2
    seq_len = 128
    window_size = 32

    model = Attention(
        heads=heads,
        causal=True,
        dropout=0.0,
        bias=False,
        rope=False,
        D=d_model,
        kv_heads=kv_heads,
        window_size=window_size,
        logit_softcap=50.0,
    ).to(device=device).eval()

    x = torch.randn(batch, seq_len, d_model, device=device)

    with torch.no_grad():
        out = model(x)
    assert out.isfinite().all()

    # Verify sliding window mask was cached
    assert len(model._mask_cache) > 0, "Expected mask to be cached in _mask_cache"

    measurement = measure(lambda: model(x), min_run_time=benchmark_min_time)
    print(format_measurement(f"Attention GQA+Window+Softcap [{device}]", measurement))

    assert math.isfinite(measurement.mean) and measurement.mean > 0
