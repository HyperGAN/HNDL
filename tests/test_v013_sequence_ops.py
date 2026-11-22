"""Regression, correctness, and numerical parity tests for v0.13 sequence & attention operators."""

import copy

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.inference import InferenceCache
from hndl.operators._attention_backend import prepare_attention
from hndl.operators._packing import prepare_packed
from hndl.operators.attention import Attention, _reference as attn_reference
from hndl.operators.cross_attention import CrossAttention
from hndl.operators.gated_delta_attention import GatedDeltaAttention, _reference as gda_reference
from hndl.operators.joint_attention import JointAttention
from hndl.operators.latent_attention import LatentAttention, _reference as mla_reference
from hndl.operators.mamba2 import Mamba2, _reference as mamba2_reference
from hndl.operators.rope import _cached_rope_tables, compute_rope_tables
from hndl.operators.transformer_block import TransformerBlock
from hndl.torch import build, network

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


# ---------------------------------------------------------------------------
# 1. GatedDeltaAttention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 17, 64, 65, 130])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_gda_channel_factored_forward_and_backward_parity(device, seq_len, chunk_size, dtype):
    if device == "cpu" and dtype == torch.bfloat16:
        pytest.skip("bfloat16 parity checked on CUDA")
    torch.manual_seed(42)
    layer = GatedDeltaAttention(
        heads=2, head_dim=16, decay_kind="channel", chunk_size=chunk_size, conv_size=4, D=32
    ).to(device=device, dtype=dtype)
    ref_fn = gda_reference(layer)

    x = torch.randn(2, seq_len, 32, device=device, dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()

    actual = layer(x)
    expected = ref_fn(xr)

    tol = {
        torch.float64: (1e-8, 1e-9),
        torch.float32: (2e-4, 2e-5),
        torch.bfloat16: (5e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])

    grad_out = torch.randn_like(actual)
    (actual * grad_out).sum().backward()
    actual_grads = [x.grad.clone()] + [p.grad.clone() for p in layer.parameters() if p.grad is not None]
    layer.zero_grad()

    (expected * grad_out).sum().backward()
    expected_grads = [xr.grad.clone()] + [p.grad.clone() for p in layer.parameters() if p.grad is not None]

    grad_tol = {
        torch.float64: (1e-7, 1e-8),
        torch.float32: (5e-4, 1e-4),
        torch.bfloat16: (8e-2, 4e-2),
    }[dtype]
    for ag, eg in zip(actual_grads, expected_grads):
        torch.testing.assert_close(ag, eg, rtol=grad_tol[0], atol=grad_tol[1])


@pytest.mark.parametrize("a_val", [5.0, 10.0, 20.0])
def test_gda_extreme_decay_fallback_parity_and_finite_grads(device, a_val):
    torch.manual_seed(7)
    layer = GatedDeltaAttention(
        heads=2, head_dim=16, decay_kind="channel", chunk_size=32, conv_size=2, D=32
    ).to(device)
    with torch.no_grad():
        layer.A_log.fill_(a_val)
    ref_fn = gda_reference(layer)

    x = torch.randn(2, 70, 32, device=device, requires_grad=True)
    actual = layer(x)
    expected = ref_fn(x.detach())
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    actual.square().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    for p in layer.parameters():
        if p.grad is not None:
            assert p.grad.isfinite().all()


@pytest.mark.parametrize("decay_kind", ["channel", "head"])
def test_gda_t1_fast_path_parity_and_autograd(device, decay_kind):
    torch.manual_seed(11)
    layer = GatedDeltaAttention(
        heads=2, head_dim=8, decay_kind=decay_kind, chunk_size=16, conv_size=4, D=16
    ).to(device)
    ref_fn = gda_reference(layer)

    # Standalone t=1 with autograd
    x1 = torch.randn(2, 1, 16, device=device, requires_grad=True)
    x1r = x1.detach().clone().requires_grad_()
    out1 = layer(x1)
    exp1 = ref_fn(x1r)
    torch.testing.assert_close(out1, exp1, rtol=1e-6, atol=1e-6)
    out1.square().sum().backward()
    exp1.square().sum().backward()
    torch.testing.assert_close(x1.grad, x1r.grad, rtol=1e-5, atol=1e-5)

    # Incremental single-token decode vs full sequence
    layer.eval()
    x_full = torch.randn(2, 19, 16, device=device)
    cache = InferenceCache(max_seq_len=32)
    with torch.no_grad():
        expected = layer(x_full)
        stepped = torch.cat(
            [layer(x_full[:, i : i + 1], cache=cache, node_id="gda") for i in range(19)],
            dim=1,
        )
    torch.testing.assert_close(stepped, expected, rtol=2e-4, atol=2e-5)


def test_gda_zero_length_conv_and_divisibility_guard(device):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divide"):
        GatedDeltaAttention(heads=5, head_dim=0, D=32)

    layer = GatedDeltaAttention(heads=2, head_dim=8, conv_size=4, chunk_size=8, D=16).to(device)
    x_t0 = torch.randn(2, 0, 16, device=device, requires_grad=True)
    out_t0 = layer(x_t0)
    assert out_t0.shape == (2, 0, 16)
    out_t0.sum().backward()
    assert x_t0.grad is not None and x_t0.grad.shape == (2, 0, 16)


def test_gda_preserves_conv_hooks_standard_and_cached(device):
    torch.manual_seed(3)
    layer = GatedDeltaAttention(heads=2, head_dim=8, conv_size=4, chunk_size=8, D=16).to(device).eval()
    ref_layer = copy.deepcopy(layer)
    x = torch.randn(2, 10, 16, device=device)

    hits = []
    h_q = layer.conv_q.register_forward_hook(lambda *_: hits.append("q"))
    h_k = layer.conv_k.register_forward_hook(lambda *_: hits.append("k"))
    h_v = layer.conv_v.register_forward_hook(lambda *_: hits.append("v"))
    try:
        with torch.no_grad():
            out_std = layer(x)
            torch.testing.assert_close(out_std, ref_layer(x), rtol=1e-5, atol=1e-6)
        assert hits == ["q", "k", "v"]

        hits.clear()
        cache_hooked = InferenceCache(max_seq_len=16)
        cache_plain = InferenceCache(max_seq_len=16)
        with torch.no_grad():
            y1 = layer(x[:, :4], cache=cache_hooked, node_id="g")
            y2 = layer(x[:, 4:5], cache=cache_hooked, node_id="g")
            y3 = layer(x[:, 5:], cache=cache_hooked, node_id="g")
            r1 = ref_layer(x[:, :4], cache=cache_plain, node_id="g")
            r2 = ref_layer(x[:, 4:5], cache=cache_plain, node_id="g")
            r3 = ref_layer(x[:, 5:], cache=cache_plain, node_id="g")
        torch.testing.assert_close(torch.cat([y1, y2, y3], dim=1), torch.cat([r1, r2, r3], dim=1), rtol=1e-5, atol=1e-6)
        assert hits == ["q", "k", "v"] * 3
        assert cache_hooked.allocated_bytes == cache_plain.allocated_bytes
    finally:
        h_q.remove()
        h_k.remove()
        h_v.remove()


def test_gda_channel_compile_fullgraph():
    layer = GatedDeltaAttention(heads=2, head_dim=8, decay_kind="channel", chunk_size=8, D=16)
    ref = copy.deepcopy(layer)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    x = torch.randn(2, 18, 16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = compiled(x)
    expected = ref(xr)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Mamba2
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="float16 overflow test on CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mamba2_reduced_precision_large_a_log_no_overflow(dtype):
    torch.manual_seed(9)
    layer = Mamba2(d_model=32, d_state=8, expand=2, heads=4, head_dim=16, chunk_size=8).to(
        device="cuda:0", dtype=dtype
    )
    with torch.no_grad():
        # exp(12.0) = 162,754 > float16 max (65,504); must not overflow to Inf/NaN.
        layer.A_log.fill_(12.0)
    x = torch.randn(2, 17, 32, device="cuda:0", dtype=dtype)
    with torch.no_grad():
        actual = layer(x)
        expected = mamba2_reference(layer)(x)
    assert actual.isfinite().all()
    assert expected.isfinite().all()
    tol = (5e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 5e-3)
    torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])


@pytest.mark.parametrize("seq_len", [1, 7, 16, 17, 48, 65])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_mamba2_vectorized_ssd_forward_and_backward_parity(device, seq_len, chunk_size):
    torch.manual_seed(19)
    layer = Mamba2(d_model=16, d_state=8, expand=2, heads=2, head_dim=16, chunk_size=chunk_size).to(device)
    ref_fn = mamba2_reference(layer)

    x = torch.randn(2, seq_len, 16, device=device, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = layer(x)
    expected = ref_fn(xr)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    grad_out = torch.randn_like(actual)
    (actual * grad_out).sum().backward()
    actual_grads = [x.grad.clone()] + [p.grad.clone() for p in layer.parameters() if p.grad is not None]
    layer.zero_grad()

    (expected * grad_out).sum().backward()
    expected_grads = [xr.grad.clone()] + [p.grad.clone() for p in layer.parameters() if p.grad is not None]
    for ag, eg in zip(actual_grads, expected_grads):
        torch.testing.assert_close(ag, eg, rtol=5e-4, atol=1e-4)


def test_mamba2_gradcheck_double():
    torch.manual_seed(21)
    layer = Mamba2(d_model=4, d_state=4, expand=2, heads=2, head_dim=4, d_conv=2, chunk_size=3).double()
    x = torch.randn(1, 7, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)


def test_mamba2_preserves_conv1d_hooks_during_cached_decode(device):
    torch.manual_seed(5)
    layer = Mamba2(d_model=16, d_state=4, expand=2, heads=4, head_dim=8, d_conv=4, chunk_size=4).to(device).eval()
    ref_layer = copy.deepcopy(layer)
    x = torch.randn(2, 11, 16, device=device)

    hits = []
    hook = layer.conv1d.register_forward_hook(lambda *_: hits.append(1))
    try:
        cache_h = InferenceCache(max_seq_len=16)
        cache_r = InferenceCache(max_seq_len=16)
        with torch.no_grad():
            full = ref_layer(x)
            parts = [
                layer(x[:, :4], cache=cache_h, node_id="m"),
                layer(x[:, 4:5], cache=cache_h, node_id="m"),
                layer(x[:, 5:], cache=cache_h, node_id="m"),
            ]
            ref_layer(x[:, :4], cache=cache_r, node_id="m")
            ref_layer(x[:, 4:5], cache=cache_r, node_id="m")
            ref_layer(x[:, 5:], cache=cache_r, node_id="m")
        torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=2e-4, atol=2e-5)
        assert len(hits) == 3
        assert cache_h.allocated_bytes == cache_r.allocated_bytes
    finally:
        hook.remove()


# ---------------------------------------------------------------------------
# 3. LatentAttention
# ---------------------------------------------------------------------------


def test_mla_absorb_unabsorb_load_state_dict_and_graph_module(device):
    torch.manual_seed(13)
    model = network(
        "latent_attention(4, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
        device=device,
    ).eval()
    layer = model[0]
    x = torch.randn(2, 8, 16, device=device)

    with torch.no_grad():
        expected = model(x)

    # Calling absorb_weights inside inference_mode must not create inference tensors or break GraphModule._verify_state.
    with torch.inference_mode():
        layer.absorb_weights()
    assert layer.is_weight_absorbed
    assert not layer._absorbed_W_uk.is_inference()
    assert not layer._absorbed_W_o.is_inference()
    assert layer._absorbed_W_uk.grad_fn is None
    assert layer._absorbed_W_o.grad_fn is None

    with torch.no_grad():
        torch.testing.assert_close(model(x), expected, rtol=1e-4, atol=1e-5)

    # Training / backward pass with absorbed weights still computes exact parameter gradients.
    layer.train()
    xg = x.detach().clone().requires_grad_()
    out_g = layer(xg)
    out_g.square().sum().backward()
    assert xg.grad is not None and xg.grad.isfinite().all()
    assert layer.k_up.weight.grad is not None and layer.v_up.weight.grad is not None
    assert layer.o_proj.weight.grad is not None
    layer.zero_grad()
    layer.eval()

    # load_state_dict must automatically refresh absorbed weights.
    new_state = {k: v.clone() * 0.5 + 0.1 for k, v in layer.state_dict().items()}
    layer.load_state_dict(new_state)
    with torch.no_grad():
        after_load_absorbed = layer(x)
        layer.unabsorb_weights()
        assert not layer.is_weight_absorbed
        assert layer._absorbed_W_uk is None and layer._absorbed_W_o is None
        after_load_standard = layer(x)
    torch.testing.assert_close(after_load_absorbed, after_load_standard, rtol=1e-4, atol=1e-5)


def test_mla_2d_axial_rope_and_divisibility_validation(device):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divide"):
        LatentAttention(heads=5, kv_lora_rank=8, qk_nope_head_dim=0, D=16)

    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divisible by 4"):
        resolve(
            "latent_attention(2, 8, qk_nope_head_dim=4, qk_rope_head_dim=6, spatial_shape=(2, 2))",
            input_shape=("B", 4, 8),
            output_shape=("B", 4, 8),
        )

    with pytest.raises(HNDLError, match="E_CONSTRAINT.*covers 4 tokens"):
        resolve(
            "latent_attention(2, 8, qk_nope_head_dim=4, qk_rope_head_dim=4, spatial_shape=(2, 2))",
            input_shape=("B", 6, 8),
            output_shape=("B", 6, 8),
        )

    layer = LatentAttention(
        heads=2, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, spatial_shape=(2, 3), D=8
    ).to(device)
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*covers 6 tokens"):
        layer(torch.randn(1, 5, 8, device=device))

    x_ok = torch.randn(2, 6, 8, device=device)
    torch.testing.assert_close(layer(x_ok), mla_reference(layer)(x_ok), rtol=1e-4, atol=1e-5)


def test_mla_packing_and_autocast(device):
    torch.manual_seed(17)
    layer = LatentAttention(
        heads=4, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, causal=True, D=32
    ).to(device)
    ref = copy.deepcopy(layer)
    prepare_packed(layer)

    x = torch.randn(2, 12, 32, device=device, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = layer(x)
    expected = ref(xr)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-4, atol=2e-5)

    # Cast clears packed cache and remains accurate
    layer.double()
    assert not getattr(layer, "_hndl_packed_layouts", {})
    with torch.no_grad():
        torch.testing.assert_close(layer(x.detach().double()), ref.double()(x.detach().double()), rtol=1e-8, atol=1e-9)

    # Autocast dtype correctness (standard and absorbed)
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    layer_ac = LatentAttention(
        heads=4, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, causal=True, D=32
    ).to(device).eval()
    x_ac = torch.randn(2, 8, 32, device=device)
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
        out_std = layer_ac(x_ac)
        layer_ac.absorb_weights()
        out_abs = layer_ac(x_ac)
    assert out_std.isfinite().all() and out_abs.isfinite().all()
    torch.testing.assert_close(out_abs.float(), out_std.float(), rtol=3e-2, atol=3e-2)


# ---------------------------------------------------------------------------
# 4. Attention
# ---------------------------------------------------------------------------


def test_attention_bounded_mask_cache_and_causal_inference_mode(device):
    layer = Attention(
        heads=4, causal=True, dropout=0.0, bias=False, rope=False, D=32, window_size=4
    ).to(device)
    with torch.no_grad():
        for length in (8, 16, 32, 12, 24, 6):
            x = torch.randn(1, length, 32, device=device)
            actual = layer(x)
            expected = attn_reference(layer)(x)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    # Bounded to 1 entry per (window_size, causal, device, dtype)
    assert len(layer._mask_cache) == 1
    cached_mask = next(iter(layer._mask_cache.values()))
    assert cached_mask.shape == (32, 32)

    # _get_causal_bias inside inference_mode must not break subsequent training
    rpb_layer = Attention(
        heads=2, causal=True, dropout=0.0, bias=True, rope=False, D=8,
        relative_position_bias=True, spatial_shape=(2, 3),
    ).to(device)
    x6 = torch.randn(2, 6, 8, device=device)
    with torch.inference_mode():
        rpb_layer.eval()(x6)
    rpb_layer.train()
    x6g = x6.detach().clone().requires_grad_()
    rpb_layer(x6g).square().sum().backward()
    assert x6g.grad is not None and x6g.grad.isfinite().all()


def test_attention_2d_axial_rope_validation_and_autocast(device):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divisible by 4"):
        resolve(
            "attention(2, rope=True, head_dim=6, spatial_shape=(2, 2))",
            input_shape=("B", 4, 12),
            output_shape=("B", 4, 12),
        )
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*covers 4 tokens"):
        resolve(
            "attention(2, rope=True, head_dim=8, spatial_shape=(2, 2))",
            input_shape=("B", 6, 16),
            output_shape=("B", 6, 16),
        )
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divisible by 4"):
        Attention(heads=2, causal=False, dropout=0.0, bias=False, rope=True, D=12, spatial_shape=(2, 2))

    layer = Attention(
        heads=2, causal=False, dropout=0.0, bias=False, rope=True, D=16, spatial_shape=(2, 3)
    ).to(device)
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*covers 6 tokens"):
        layer(torch.randn(1, 5, 16, device=device))

    x_ok = torch.randn(2, 6, 16, device=device)
    torch.testing.assert_close(layer(x_ok), attn_reference(layer)(x_ok), rtol=1e-4, atol=1e-5)

    # Autocast with RoPE + window_size + relative_position_bias
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    layer_ac = Attention(
        heads=2, causal=True, dropout=0.0, bias=True, rope=True, D=16,
        relative_position_bias=True, spatial_shape=(2, 3), window_size=4,
    ).to(device).eval()
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
        out = layer_ac(x_ok)
    assert out.isfinite().all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("softcap", [0.0, 2.0])
def test_attention_cached_t1_skips_per_token_block_mask_compilation(softcap):
    torch.manual_seed(31)
    layer = Attention(
        heads=4, causal=True, dropout=0.0, bias=False, rope=True, D=64,
        kv_heads=2, head_dim=16, window_size=8, logit_softcap=softcap,
    ).cuda().eval()
    ref = copy.deepcopy(layer)
    prepare_attention(layer, backend="auto")

    x = torch.randn(2, 24, 64, device="cuda")
    cache = InferenceCache(max_seq_len=32)
    with torch.no_grad():
        expected = ref(x)
        # Prefill 8 tokens, then 16 single-token decode steps
        out_parts = [layer(x[:, :8], cache=cache, node_id="a")]
        masks_after_prefill = len(layer._hndl_attention_masks)
        for i in range(8, 24):
            out_parts.append(layer(x[:, i : i + 1], cache=cache, node_id="a"))
        actual = torch.cat(out_parts, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-4)
    # No new block masks compiled during the 16 single-token decode steps!
    assert len(layer._hndl_attention_masks) == masks_after_prefill


# ---------------------------------------------------------------------------
# 5. RoPE
# ---------------------------------------------------------------------------


def test_rope_2d_axial_ntk_exponent_and_validation(device):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divisible by 4"):
        compute_rope_tables(4, 6, torch.device(device), torch.float32, spatial_shape=(2, 2))

    # Verify 2D axial NTK uses d_axis / max(d_axis - 2, 1)
    head_dim = 16
    d_axis = head_dim // 2
    theta = 10000.0
    scale = 2.0
    cos, sin = compute_rope_tables(
        6, head_dim, torch.device(device), torch.float64,
        theta=theta, rope_scaling="ntk", rope_scale=scale, spatial_shape=(2, 3),
    )
    theta_ntk = theta * (scale ** (d_axis / max(d_axis - 2, 1)))
    exp = torch.arange(0, d_axis, 2, device=device, dtype=torch.float64) / d_axis
    inv_freq = theta_ntk ** (-exp)
    idx = torch.arange(6, device=device, dtype=torch.float64)
    ang = torch.cat(((idx // 3)[:, None] * inv_freq[None, :], (idx % 3)[:, None] * inv_freq[None, :]), dim=-1)
    expected_cos = torch.cat((ang.cos(), ang.cos()), dim=-1)[None, None]
    expected_sin = torch.cat((ang.sin(), ang.sin()), dim=-1)[None, None]
    torch.testing.assert_close(cos, expected_cos, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sin, expected_sin, rtol=1e-12, atol=1e-12)


def test_rope_cached_tables_compile_start_slice():
    cache = {}

    @torch.compile(backend="eager", fullgraph=True)
    def fn(x):
        cos, sin = _cached_rope_tables(cache, 10, 8, x.device, x.dtype, 10000.0, "none", 1.0, (0, 0), start=4)
        return x * cos + sin

    x = torch.randn(1, 1, 6, 8)
    actual = fn(x)
    full_cos, full_sin = compute_rope_tables(10, 8, x.device, x.dtype)
    expected = x * full_cos[..., 4:10, :] + full_sin[..., 4:10, :]
    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# 6. TransformerBlock
# ---------------------------------------------------------------------------


def test_transformer_block_explicit_head_dim_and_node_id_guard(device):
    # D=30 is not divisible by heads=4, which is valid when head_dim=8 is explicit!
    plan = resolve(
        "transformer_block(4, head_dim=8, rope=True, causal=True)",
        input_shape=("B", 6, 30),
        output_shape=("B", 6, 30),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 6, 30, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 6, 30) and out.isfinite().all()

    # Odd explicit head_dim with rope=True must still be rejected
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*even"):
        resolve(
            "transformer_block(4, head_dim=7, rope=True)",
            input_shape=("B", 6, 32),
            output_shape=("B", 6, 32),
        )

    # Empty or None node_id with cache must raise E_RUNTIME
    block = TransformerBlock(
        heads=4, mlp_ratio=2, activation="gelu", norm="rms_norm", causal=True,
        dropout=0.0, layer_scale=False, layer_scale_init=1e-5, eps=1e-5, D=16,
    ).to(device).eval()
    cache = InferenceCache(max_seq_len=16)
    with torch.no_grad():
        for bad_id in (None, ""):
            with pytest.raises(HNDLError, match="E_RUNTIME.*node_id"):
                block(torch.randn(1, 2, 16, device=device), cache=cache, node_id=bad_id)


# ---------------------------------------------------------------------------
# 7. CrossAttention & JointAttention
# ---------------------------------------------------------------------------


def test_cross_attention_has_no_dead_mask_cache():
    layer = CrossAttention(heads=2, bias=True, dropout=0.0, D=16, D_ctx=16)
    assert not hasattr(layer, "_mask_cache")


def test_joint_attention_divisibility_and_packing(device):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divide"):
        JointAttention(heads=5, head_dim=0, D_x=32, D_c=32)

    torch.manual_seed(23)
    layer = JointAttention(heads=4, head_dim=8, qk_norm="rms_norm", D_x=32, D_c=16).to(device)
    ref = copy.deepcopy(layer)
    prepare_packed(layer)

    x = torch.randn(2, 6, 32, device=device, requires_grad=True)
    ctx = torch.randn(2, 10, 16, device=device, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    ctxr = ctx.detach().clone().requires_grad_()

    out, ctx_out = layer(x, ctx)
    out_r, ctx_out_r = ref(xr, ctxr)
    torch.testing.assert_close(out, out_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ctx_out, ctx_out_r, rtol=1e-4, atol=1e-5)

    (out.square().sum() + ctx_out.square().sum()).backward()
    (out_r.square().sum() + ctx_out_r.square().sum()).backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(ctx.grad, ctxr.grad, rtol=2e-4, atol=2e-5)

    # Cast clears packed layouts
    layer.double()
    assert not getattr(layer, "_hndl_packed_layouts", {})
    with torch.no_grad():
        o64, c64 = layer(x.detach().double(), ctx.detach().double())
        o64_r, c64_r = ref.double()(x.detach().double(), ctx.detach().double())
        torch.testing.assert_close(o64, o64_r, rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(c64, c64_r, rtol=1e-8, atol=1e-9)
