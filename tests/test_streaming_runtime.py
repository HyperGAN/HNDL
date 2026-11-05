"""Whole-graph incremental parity and sequence-only contract relaxation."""

import pytest
import torch

from hndl.errors import HNDLError
from hndl.inference import InferenceCache as PublicCache
from hndl.torch import InferenceCache, network


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
SOURCES = {
    "gpt": "embedding(32,16); pos_embed(32); transformer_block(4,causal=True); transformer_block(4,causal=True); linear(32)",
    "llama": "rms_norm(); attention(4,kv_heads=2,rope=True,causal=True); swiglu(32); linear(16)",
    "hybrid": "mamba2(16,d_state=8,heads=4,head_dim=8,chunk_size=4); gated_delta_attention(4,head_dim=4,chunk_size=4); latent_attention(4,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,causal=True)",
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("kind", SOURCES)
def test_full_graph_matches_partitioned_incremental_execution(device, prepared, kind):
    tokens = kind == "gpt"
    model = network(SOURCES[kind], input_shape=("B", 13) if tokens else ("B", 13, 16),
                    input_dtype="int64" if tokens else "float32",
                    output_shape=("B", 13, 32 if tokens else 16), device=device).eval()
    x = torch.randint(0, 32, (2, 13), device=device) if tokens else torch.randn(2, 13, 16, device=device)
    executor = model.prepare() if prepared else model
    cache = InferenceCache(max_seq_len=16)
    assert InferenceCache is PublicCache
    with torch.no_grad():
        expected = model(x)
        actual = torch.cat([executor(part, cache=cache) for part in x.split((5, 1, 3, 4), dim=1)], dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        assert cache.allocated_bytes > 0
        assert all(cache.position(key) == 13 for key in cache.keys())
        cache.reset()
        torch.testing.assert_close(executor(x, cache=cache), expected, rtol=2e-4, atol=2e-5)


def test_hot_stream_signature_uses_generated_program(monkeypatch):
    model = network("attention(4,causal=True); linear(16)", input_shape=("B", 8, 16),
                    output_shape=("B", 8, 16), device="cpu").eval()
    cache = InferenceCache(16)
    x = torch.randn(2, 1, 16)
    with torch.no_grad():
        model(x, cache=cache)
        monkeypatch.setattr(model, "_check", lambda *args: pytest.fail("hot streaming path revalidated a port"))
        model(x, cache=cache)
        assert cache.position("n0") == 2
    assert "del " in model._stream_executor.source_code
    assert "_stream_executor" not in model._modules


@pytest.mark.parametrize("operation", ["softmax(dim=1)", "l2_norm(dim=1)", "pool_tokens('mean')"])
def test_time_mixing_operators_are_rejected(operation):
    shape = ("B", 8, 16) if "pool_tokens" not in operation else ("B", 16)
    model = network("attention(4,causal=True); " + operation,
                    input_shape=("B", 8, 16), output_shape=shape, device="cpu").eval()
    cache = InferenceCache()
    with torch.no_grad(), pytest.raises(HNDLError, match="incremental|sequence"):
        model(torch.randn(2, 1, 16), cache=cache)


def test_failed_call_requires_reset_and_does_not_relax_feature_axes():
    model = network("attention(4,causal=True); linear(8)", input_shape=("B", 8, 16),
                    output_shape=("B", 8, 8), device="cpu").eval()
    cache = InferenceCache()
    with torch.no_grad():
        with pytest.raises(HNDLError, match="expected shape"):
            model(torch.randn(2, 1, 20), cache=cache)
        with pytest.raises(HNDLError, match="previous cached call failed"):
            model(torch.randn(2, 1, 16), cache=cache)
        cache.reset()
        assert model(torch.randn(2, 1, 16), cache=cache).shape == (2, 1, 8)
        model[1].weight.add_(1)
        with pytest.raises(HNDLError, match="parameters"):
            model(torch.randn(2, 1, 16), cache=cache)


def test_cache_cannot_bypass_an_image_contract():
    model = network("conv(4,kernel_size=3,padding=1)", input_shape=("B", 3, 8, 8),
                    output_shape=("B", 4, 8, 8), device="cpu").eval()
    with torch.no_grad(), pytest.raises(HNDLError, match="rank-2|rank-3"):
        model(torch.randn(2, 3, 5, 8), cache=InferenceCache())


def test_cached_graph_error_during_forward_invalidates_session():
    model = network("attention(4,causal=True); pos_embed(8)", input_shape=("B", 8, 16),
                    output_shape=("B", 8, 16), device="cpu").eval()
    cache = InferenceCache(16)
    with torch.no_grad():
        model(torch.randn(2, 8, 16), cache=cache)
        with pytest.raises(HNDLError, match="max_len"):
            model(torch.randn(2, 1, 16), cache=cache)
        with pytest.raises(HNDLError, match="previous cached call failed"):
            model(torch.randn(2, 1, 16), cache=cache)


def test_token_id_cast_can_preserve_rank_two_sequence_axis():
    model = network("cast('int64'); embedding(32,16); attention(4,causal=True)",
                    input_shape=("B", 5), input_dtype="int32",
                    output_shape=("B", 5, 16), device="cpu").eval()
    x = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    with torch.no_grad():
        expected = model(x)
        cache = InferenceCache(8)
        actual = torch.cat([model(part, cache=cache) for part in x.split((2, 1, 2), dim=1)], dim=1)
        torch.testing.assert_close(actual, expected)
