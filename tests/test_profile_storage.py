"""Cross-check static storage estimates against the corresponding runtime objects."""

import pytest
import torch

from hndl import resolve
from hndl.inference import InferenceCache
from hndl.torch import build


def test_profile_respects_affine_and_mixed_dtype_liveness():
    shape = {"input_shape": ("B", 16), "output_shape": ("B", 16)}
    plan = resolve('layer_norm(affine=False); cast("float64")', **shape)
    profile = plan.profile(batch_size=2)
    model = build(plan, device="cpu")
    tensor = torch.ones(2, 16)
    result = model(tensor)
    assert profile.total_parameters == sum(p.numel() for p in model.parameters()) == 0
    assert profile.nodes[-1].activation_bytes == result.nbytes == 256
    assert profile.peak_activation_bytes == result.nbytes + tensor.nbytes
    cast_ids = resolve('cast("bool")', input_dtype="int64", **shape).profile(2)
    assert cast_ids.nodes[0].activation_bytes == 32
    assert cast_ids.peak_activation_bytes == 32 * (8 + 1)


def test_profile_counts_unique_nested_and_transitive_parameters():
    shape = {"input_shape": ("B", 3, 8), "output_shape": ("B", 3, 8)}
    source = ('attention(2, name="a")\n'
              'attention(2, name="b"{first})\n'
              'attention(2, name="c"{second})')
    tied = resolve(source.format(first=', tie={"q_proj.weight": "a.k_proj.weight"}',
                                 second=', tie={"o_proj.weight": "b.q_proj.weight"}'), **shape)
    plain = resolve(source.format(first="", second=""), **shape)
    profile = tied.profile()
    actual = sum(p.numel() for p in build(tied, device="cpu").parameters())
    assert actual == profile.total_parameters == profile.active_parameters
    assert sum(node.total_parameters for node in profile.nodes) == plain.profile().total_parameters
    assert plain.diff(tied).param_delta == -128


@pytest.mark.parametrize(("source", "dtype", "absorbed"), [
    ("attention(4, causal=True, kv_heads=2, window_size=3)", "float32", False),
    ("transformer_block(4, causal=True, window_size=3)", "float32", False),
    ("attention(4, causal=True, window_size=32)", "float32", False),
    ("latent_attention(4, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4)",
     "float32", False),
    ("latent_attention(4, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4)",
     "float32", True),
    ("mamba2(d_model=16, d_state=4, heads=4, head_dim=8)", "bfloat16", False),
    ("gated_delta_attention(4, head_dim=4)", "bfloat16", False),
])
def test_profile_cache_estimate_matches_session(source, dtype, absorbed):
    plan = resolve(source, input_shape=("B", 8, 16), output_shape=("B", 8, 16), dtype=dtype)
    model = build(plan, device="cpu").eval()
    profile = plan.profile(2, cache_capacity=12, absorbed_mla={"n0"} if absorbed else False)
    cache = InferenceCache(max_seq_len=12)
    with torch.no_grad():
        if absorbed:
            with model.allow_state_mutation():
                model[0].absorb_weights()
            assert plan.profile(2, cache_capacity=12, absorbed_mla=True) == profile
        model(torch.ones(2, 1, 16, dtype=getattr(torch, dtype)), cache=cache)
    assert profile.kv_cache_bytes == cache.allocated_bytes
    assert profile.cache_capacity == 12
    assert profile.absorbed_mla_nodes == (("n0",) if absorbed else ())


def test_profile_rejects_invalid_cache_assumptions():
    plan = resolve("latent_attention(2, kv_lora_rank=4, bias=True)",
                   input_shape=("B", 8, 16), output_shape=("B", 8, 16))
    with pytest.raises(ValueError, match="bias=False"):
        plan.profile(absorbed_mla=True)
    with pytest.raises(ValueError, match="latent_attention nodes"):
        plan.profile(absorbed_mla=["unknown"])
    for capacity in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="cache_capacity"):
            plan.profile(cache_capacity=capacity)
