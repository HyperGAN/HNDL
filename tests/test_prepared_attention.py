"""Prepared attention uses packed projections and fused masks without drift."""

import copy

import pytest
import torch

from hndl.inference import InferenceCache
from hndl.torch import network
from hndl.operators._attention_backend import prepare_attention
from hndl.operators._packing import prepare_packed


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def layer_for(source, device, length=9):
    return network(source, input_shape=("B", length, 32),
                   output_shape=("B", length, 32), device=device)[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source", [
    "attention(4, causal=True, rope=True, kv_heads=2, qk_norm='rms_norm')",
    "attention(4, causal=True, window_size=4, logit_softcap=2.0)",
    "gated_delta_attention(4, head_dim=8, chunk_size=4, decay_kind='head')",
    "gated_delta_attention(4, head_dim=8, chunk_size=4, decay_kind='channel')",
])
def test_packed_qkv_preserves_gradients_optimizer_and_state(source, device):
    torch.manual_seed(12)
    layer = layer_for(source, device)
    reference = copy.deepcopy(layer)
    original_ids = tuple(id(p) for p in layer.parameters())
    state_keys = tuple(layer.state_dict())
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    prepare_packed(layer)
    assert tuple(id(p) for p in layer.parameters()) == original_ids
    assert tuple(layer.state_dict()) == state_keys
    for _ in range(2):
        x = torch.randn(2, 9, 32, device=device, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual, expected = layer(x), reference(xr)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad, rtol=2e-3, atol=2e-4)
        for parameter, expected_parameter in zip(layer.parameters(), reference.parameters()):
            if parameter.grad is not None or expected_parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=2e-3, atol=2e-4)
        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.zero_grad()


@pytest.mark.parametrize("device", DEVICES)
def test_packed_attention_keeps_projection_hooks_and_clears_layout_on_cast(device):
    layer = layer_for("attention(4, causal=True)", device).eval()
    reference = copy.deepcopy(layer)
    prepare_packed(layer)
    seen = []
    hook = layer.q_proj.register_forward_hook(lambda *args: seen.append(1))
    x = torch.randn(2, 9, 32, device=device)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), reference(x))
    assert seen == [1]
    hook.remove()
    assert layer._hndl_packed_layouts
    layer.double()
    assert not getattr(layer, "_hndl_packed_layouts", {})
    with torch.no_grad():
        torch.testing.assert_close(layer(x.double()), reference.double()(x.double()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source", [
    "attention(4, causal=True, rope=True, kv_heads=2, window_size=4, logit_softcap=2.0)",
    "gated_delta_attention(4, head_dim=8, chunk_size=4, decay_kind='channel')",
])
def test_packed_qkv_preserves_incremental_state(source, device):
    layer = layer_for(source, device).eval()
    reference = copy.deepcopy(layer)
    prepare_packed(layer)
    cache = InferenceCache(max_seq_len=9)
    x = torch.randn(2, 9, 32, device=device)
    with torch.no_grad():
        expected = reference(x)
        actual = torch.cat([layer(x[:, :4], cache=cache, node_id="layer"),
                            layer(x[:, 4:5], cache=cache, node_id="layer"),
                            layer(x[:, 5:], cache=cache, node_id="layer")], dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert cache.position("layer") == 9


def test_attention_cpu_fused_backend_falls_back_without_changing_math():
    layer = layer_for("attention(4, causal=True, window_size=4, logit_softcap=2.0)", "cpu").eval()
    reference = copy.deepcopy(layer)
    prepare_attention(layer, backend="auto")
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), reference(x))
    assert layer._hndl_attention_backend == "reference"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_attention_full_and_cached_positions_avoid_dense_window_mask():
    source = "attention(4, head_dim=16, causal=True, rope=True, kv_heads=2, window_size=17, logit_softcap=2.0)"
    layer = layer_for(source, "cuda:0", length=33).eval()
    reference = copy.deepcopy(layer)
    prepare_packed(layer)
    prepare_attention(layer, backend="flex")
    x = torch.randn(2, 33, 32, device="cuda:0")
    cache = InferenceCache(max_seq_len=33)
    with torch.no_grad():
        expected = reference(x)
        actual = layer(x)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
        chunks = [layer(x[:, :16], cache=cache, node_id="a"),
                  layer(x[:, 16:17], cache=cache, node_id="a"),
                  layer(x[:, 17:], cache=cache, node_id="a")]
        torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=2e-3, atol=2e-4)
    assert layer._hndl_attention_backend == "flex"
    assert not layer._mask_cache
    assert len(layer._hndl_attention_masks) <= 16
    layer.cpu()
    assert not layer._hndl_attention_masks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_attention_with_packed_qkv_preserves_training_gradients():
    layer = layer_for("attention(4, head_dim=16, causal=True, kv_heads=2, window_size=5, logit_softcap=2.0)",
                      "cuda:0", length=17)
    reference = copy.deepcopy(layer)
    prepare_packed(layer)
    prepare_attention(layer, backend="flex")
    x = torch.randn(2, 17, 32, device="cuda:0", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual, expected = layer(x), reference(xr)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-3, atol=2e-4)
    for parameter, expected_parameter in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=2e-3, atol=2e-4)
    assert layer._hndl_attention_backend == "flex"
    assert not layer._mask_cache


@pytest.mark.parametrize("source", [
    "attention(4, causal=True, rope=True, kv_heads=2)",
    "gated_delta_attention(4, head_dim=8, chunk_size=4)",
])
def test_packed_attention_remains_fullgraph_compilable(source):
    layer = layer_for(source, "cpu")
    reference = copy.deepcopy(layer)
    prepare_packed(layer)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    x = torch.randn(2, 9, 32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual, expected = compiled(x), reference(xr)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-3, atol=2e-4)
