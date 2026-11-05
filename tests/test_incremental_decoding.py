"""Partitioned inference must equal a full causal pass without replaying history."""

import pytest
import torch

from hndl.errors import HNDLError
from hndl.inference import InferenceCache
from hndl.torch import network


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def make_layer(source, device, length=13, width=16):
    torch.manual_seed(12)
    model = network(source, input_shape=("B", length, width),
                    output_shape=("B", length, width), device=device).eval()
    return model[0]


def streamed(layer, x, parts, cache=None):
    cache = cache if cache is not None else InferenceCache(max_seq_len=x.shape[1])
    values, offset = [], 0
    for count in parts:
        values.append(layer(x[:, offset:offset + count], cache=cache, node_id="layer"))
        offset += count
    assert offset == x.shape[1]
    return torch.cat(values, dim=1), cache


@pytest.mark.parametrize("source", [
    "attention(4, causal=True)",
    "attention(4, causal=True, kv_heads=2, rope=True)",
    "attention(4, causal=True, kv_heads=1, rope=True, qk_norm='rms_norm')",
    "attention(4, causal=True, kv_heads=2, qk_norm='l2_norm', window_size=4)",
    "attention(4, causal=True, kv_heads=2, rope=True, logit_softcap=2.0, window_size=4)",
    "attention(4, causal=True, logit_softcap=1.0)",
    "transformer_block(4, causal=True, rope=True, norm='rms_norm', ffn_kind='swiglu')",
    "transformer_block(4, causal=True, post_norm=True, layer_scale=True)",
    "latent_attention(4, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)",
    "latent_attention(4, kv_lora_rank=8, q_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=0, v_head_dim=4, bias=True)",
    "mamba2(d_model=16, d_state=4, heads=4, head_dim=8, chunk_size=4)",
    "mamba2(d_model=16, d_state=4, heads=4, head_dim=8, chunk_size=4, d_conv=1)",
    "gated_delta_attention(4, head_dim=4, chunk_size=4, decay_kind='head')",
    "gated_delta_attention(4, head_dim=4, chunk_size=4, decay_kind='channel')",
    "gated_delta_attention(4, head_dim=4, chunk_size=4, conv_size=0)",
    "pos_embed(32)",
    "rope(4)",
    "rope(4, rope_scaling='linear', rope_scale=2.0)",
    "rope(4, rope_scaling='ntk', rope_scale=2.0)",
    "rope(4, rope_scaling='llama3', rope_scale=2.0)",
])
@pytest.mark.parametrize("parts", [(5, 1, 3, 4), (1,) * 13])
def test_partitioned_matches_full(source, parts, device):
    layer = make_layer(source, device)
    x = torch.randn(2, 13, 16, device=device)
    with torch.no_grad():
        expected = layer(x)
        actual, cache = streamed(layer, x, parts)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert cache.position("layer/attention" if source.startswith("transformer_block") else "layer") == 13


@pytest.mark.parametrize("absorbed", [False, True])
def test_mla_cache_storage_and_absorption(absorbed, device):
    layer = make_layer("latent_attention(4, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)", device)
    if absorbed:
        layer.absorb_weights()
    x = torch.randn(2, 13, 16, device=device)
    with torch.inference_mode():
        expected = layer(x)
        actual, cache = streamed(layer, x, (5, 1, 7))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    scalar_size = x.element_size()
    elements = 2 * 13 * (8 + 4) if absorbed else 2 * 4 * 13 * (8 + 4)
    assert cache.allocated_bytes == elements * scalar_size


def test_attention_reuses_fixed_storage(device):
    layer = make_layer("attention(4, causal=True, rope=True)", device)
    cache = InferenceCache(max_seq_len=16)
    with torch.no_grad():
        layer(torch.randn(2, 5, 16, device=device), cache=cache, node_id="a")
        before = {k: (t.data_ptr(), tuple(t.shape)) for k, t in cache["a"].tensors.items()}
        bytes_before = cache.allocated_bytes
        layer(torch.randn(2, 1, 16, device=device), cache=cache, node_id="a")
    assert cache.position("a") == 6
    assert before == {k: (t.data_ptr(), tuple(t.shape)) for k, t in cache["a"].tensors.items()}
    assert cache.allocated_bytes == bytes_before


@pytest.mark.parametrize("source", [
    "mamba2(d_model=16, d_state=4, heads=4, head_dim=8)",
    "gated_delta_attention(4, head_dim=4, chunk_size=4)",
])
def test_recurrent_state_memory_does_not_grow(source, device):
    layer = make_layer(source, device)
    cache = InferenceCache(max_seq_len=128)
    with torch.no_grad():
        layer(torch.randn(2, 5, 16, device=device), cache=cache, node_id="a")
        before = {k: (t.data_ptr(), tuple(t.shape)) for k, t in cache["a"].tensors.items()}
        for _ in range(3):
            layer(torch.randn(2, 1, 16, device=device), cache=cache, node_id="a")
    assert before == {k: (t.data_ptr(), tuple(t.shape)) for k, t in cache["a"].tensors.items()}
    assert cache.position("a") == 8


def test_cache_rejects_capacity_drift_and_reset(device):
    layer = make_layer("attention(4, causal=True)", device)
    cache = InferenceCache(max_seq_len=3)
    x = torch.randn(2, 2, 16, device=device)
    with torch.no_grad():
        first = layer(x, cache=cache, node_id="a")
        with pytest.raises(HNDLError, match="capacity"):
            layer(x, cache=cache, node_id="a")
        with pytest.raises(HNDLError, match="batch"):
            layer(x[:1, :1], cache=cache, node_id="a")
        cache.reset("a")
        torch.testing.assert_close(layer(x, cache=cache, node_id="a"), first)
        with torch.no_grad():
            layer.q_proj.weight.add_(0.1)
        with pytest.raises(HNDLError, match="parameters"):
            layer(x[:, :1], cache=cache, node_id="a")
        cache.reset()
        layer(x, cache=cache, node_id="a")


def test_cache_requires_inference_and_causal(device):
    layer = make_layer("attention(4, causal=True)", device)
    x = torch.randn(2, 1, 16, device=device)
    with pytest.raises(HNDLError, match="eval"):
        layer(x, cache=InferenceCache(), node_id="a")
    with torch.no_grad():
        layer.train()
        with pytest.raises(HNDLError, match="eval"):
            layer(x, cache=InferenceCache(), node_id="a")
        noncausal = make_layer("attention(4)", device)
        with pytest.raises(HNDLError, match="causal"):
            noncausal(x, cache=InferenceCache(), node_id="a")


def test_failed_graph_call_requires_explicit_reset(device):
    model = network("attention(4, causal=True)", input_shape=("B", 13, 16), output_shape=("B", 13, 16), device=device).eval()
    cache = InferenceCache()
    inputs = (torch.randn(2, 1, 16, device=device),)
    with torch.no_grad():
        cache._begin_call(model, inputs)
        cache._end_call(False)
        with pytest.raises(HNDLError, match="previous cached call failed"):
            cache._begin_call(model, inputs)
        cache.reset()
        cache._begin_call(model, inputs)
        with pytest.raises(HNDLError, match="concurrently"):
            cache._begin_call(model, inputs)
        cache._end_call(True)


def test_standalone_rope_head_major_offsets(device):
    from hndl.operators.rope import RoPE
    layer = RoPE(heads=4).to(device).eval()
    x = torch.randn(2, 4, 13, 8, device=device)
    cache = InferenceCache(max_seq_len=13)
    with torch.no_grad():
        expected = layer(x)
        actual = torch.cat([layer(x[:, :, :5], cache=cache, node_id="r"),
                            layer(x[:, :, 5:], cache=cache, node_id="r")], dim=2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("source", ["rope(4, rope_scaling='yarn', rope_scale=2.0)",
                                   "attention(4, causal=True, rope=True, rope_scaling='yarn', rope_scale=2.0)"])
def test_length_dependent_rope_is_explicitly_unsupported(source):
    layer = make_layer(source, "cpu")
    with torch.no_grad(), pytest.raises(HNDLError, match="length-independent|YaRN"):
        layer(torch.randn(2, 1, 16), cache=InferenceCache(), node_id="a")


@pytest.mark.parametrize("source", [
    "attention(4, causal=True, kv_heads=2, rope=True, qk_norm='rms_norm')",
    "latent_attention(4, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)",
    "mamba2(d_model=16, d_state=4, heads=4, head_dim=8, chunk_size=4)",
    "gated_delta_attention(4, head_dim=4, chunk_size=4, decay_kind='head')",
    "gated_delta_attention(4, head_dim=4, chunk_size=4, decay_kind='channel')",
])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_partitioned_precision(source, dtype, device):
    if device == "cpu" and dtype != torch.float64:
        pytest.skip("reduced precision qualification is on CUDA")
    layer = make_layer(source, device).to(dtype=dtype)
    x = torch.randn(2, 13, 16, device=device, dtype=dtype)
    with torch.no_grad():
        expected = layer(x)
        actual, _ = streamed(layer, x, (5, 1, 3, 4))
    tol = {torch.float64: (1e-8, 1e-9), torch.float16: (1e-2, 2e-3), torch.bfloat16: (5e-2, 1e-2)}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])


def test_cache_rejects_owner_and_dtype_drift(device):
    layer = make_layer("attention(4, causal=True)", device)
    other = make_layer("attention(4, causal=True)", device)
    cache = InferenceCache()
    x = torch.randn(2, 1, 16, device=device)
    with torch.no_grad():
        layer(x, cache=cache, node_id="a")
        with pytest.raises(HNDLError, match="owner"):
            other(x, cache=cache, node_id="a")
        layer.double()
        with pytest.raises(HNDLError, match="dtype"):
            layer(x.double(), cache=cache, node_id="a")
        cache.reset()
        layer(x.double(), cache=cache, node_id="a")


def test_cache_rejects_graph_model_and_batch_drift(device):
    model = network("attention(4, causal=True)", input_shape=("B", 13, 16), output_shape=("B", 13, 16), device=device).eval()
    other = network("attention(4, causal=True)", input_shape=("B", 13, 16), output_shape=("B", 13, 16), device=device).eval()
    cache = InferenceCache()
    inputs = (torch.randn(2, 1, 16, device=device),)
    with torch.no_grad():
        cache._begin_call(model, inputs)
        cache._end_call(True)
        with pytest.raises(HNDLError, match="model"):
            cache._begin_call(other, inputs)
        with pytest.raises(HNDLError, match="batch"):
            cache._begin_call(model, (inputs[0][:1],))
        cache.reset()
        cache._begin_call(other, inputs)
        cache._end_call(True)


def test_cache_stops_positional_overflow_before_advancing(device):
    layer = make_layer("pos_embed(13)", device)
    cache = InferenceCache(max_seq_len=32)
    with torch.no_grad():
        layer(torch.randn(2, 13, 16, device=device), cache=cache, node_id="position")
        with pytest.raises(HNDLError, match="max_len"):
            layer(torch.randn(2, 1, 16, device=device), cache=cache, node_id="position")
    assert cache.position("position") == 13


def test_graph_guards_all_parameters_and_structure_epoch(device):
    model = network("linear(16)\nattention(4, causal=True)", input_shape=("B", 13, 16),
                    output_shape=("B", 13, 16), device=device).eval()
    inputs = (torch.randn(2, 1, 16, device=device),)
    cache = InferenceCache()
    with torch.no_grad():
        cache._begin_call(model, inputs)
        cache._end_call(True)
        model[0].weight.add_(1)
        with pytest.raises(HNDLError, match="parameters"):
            cache._begin_call(model, inputs)
        cache.reset()
        cache._begin_call(model, inputs)
        cache._end_call(True)
        model._preparation_epoch = 1
        with pytest.raises(HNDLError, match="model"):
            cache._begin_call(model, inputs)


def test_partial_graph_reset_cannot_silently_lose_context(device):
    model = network("attention(4, causal=True)", input_shape=("B", 13, 16),
                    output_shape=("B", 13, 16), device=device).eval()
    inputs = (torch.randn(2, 1, 16, device=device),)
    cache = InferenceCache()
    with torch.no_grad():
        cache._begin_call(model, inputs)
        model[0](inputs[0], cache=cache, node_id="a")
        cache._end_call(True)
        cache.reset("a")
        with pytest.raises(HNDLError, match="complete session"):
            cache._begin_call(model, inputs)


def test_graph_hot_path_uses_one_guard_and_standalone_keeps_its_guard(device, monkeypatch):
    import hndl.inference as inference
    model = network("attention(4, causal=True)", input_shape=("B", 13, 16),
                    output_shape=("B", 13, 16), device=device).eval()
    cache = InferenceCache()
    inputs = (torch.randn(2, 1, 16, device=device),)
    with torch.no_grad():
        cache._begin_call(model, inputs)
        model[0](inputs[0], cache=cache, node_id="a")
        cache._end_call(True)
        original = inference._parameter_signature
        guarded = []
        def record(module):
            guarded.append(module)
            return original(module)
        monkeypatch.setattr(inference, "_parameter_signature", record)
        cache._begin_call(model, inputs)
        model[0](inputs[0], cache=cache, node_id="a")
        cache._end_call(True)
        assert guarded == [model]
        model[0].q_proj.weight.add_(0.1)
        with pytest.raises(HNDLError, match="parameters"):
            model[0](inputs[0], cache=cache, node_id="a")
        assert guarded == [model, model[0]]
