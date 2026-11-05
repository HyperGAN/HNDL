"""Specialized attention preserves scores, masks, GQA and gradients."""

import math
import inspect

import pytest
import torch
from torch import nn

from hndl.operators import _attention_backend as backend


def _reference(q, k, v, *, causal, window_size, softcap, query_offset=0):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if softcap:
        scores = softcap * (scores / softcap).tanh()
    delta = torch.arange(q.shape[-2], device=q.device)[:, None] + query_offset
    delta = delta - torch.arange(k.shape[-2], device=q.device)[None, :]
    allowed = torch.ones_like(delta, dtype=torch.bool)
    if causal:
        allowed = allowed & (delta >= 0)
    if window_size:
        allowed = allowed & ((delta < window_size) if causal else (delta.abs() < window_size))
    scores = scores.masked_fill(~allowed, -float("inf"))
    return scores.softmax(-1) @ v


def test_cpu_fallback_and_explicit_backend():
    owner = backend.prepare_attention(nn.Identity())
    q = torch.randn(1, 2, 4, 16)
    assert backend.specialized_attention(q, q, q, softcap=1.0, owner=owner) is None
    assert owner._hndl_attention_backend == "reference"
    assert owner._hndl_attention_fallback_reason == "device or dtype"
    backend.prepare_attention(owner, backend="flex")
    with pytest.raises(RuntimeError, match="FlexAttention is unavailable"):
        backend.specialized_attention(q, q, q, softcap=1.0, owner=owner)
    backend.prepare_attention(owner, backend="reference")
    assert backend.specialized_attention(q, q, q, softcap=1.0, owner=owner) is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("causal,window_size,softcap,query_offset", [
    (True, 17, 2.0, 0),
    (False, 13, 0.0, 0),
    (True, 17, 2.0, 23),
])
def test_cuda_flex_matches_reference_and_backward(causal, window_size, softcap, query_offset):
    owner = backend.prepare_attention(nn.Identity(), backend="flex")
    qlen, klen = (7, 30) if query_offset else (37, 37)
    q = torch.randn(1, 4, qlen, 16, device="cuda", requires_grad=True)
    k = torch.randn(1, 2, klen, 16, device="cuda", requires_grad=True)
    v = torch.randn(1, 2, klen, 16, device="cuda", requires_grad=True)
    options = dict(causal=causal, window_size=window_size, softcap=softcap, query_offset=query_offset)
    actual = backend.specialized_attention(q, k, v, owner=owner, **options)
    expected = _reference(q, k, v, **options)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    gradients = torch.autograd.grad(actual.square().sum(), (q, k, v))
    reference_gradients = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, b in zip(gradients, reference_gradients):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
    assert owner._hndl_attention_backend == "flex"
    assert len(owner._hndl_attention_masks) == 1
    backend.clear_attention_cache(owner)
    assert not owner._hndl_attention_masks


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_preserves_dropout_and_dense_bias():
    owner = backend.prepare_attention(nn.Identity())
    q = torch.randn(1, 2, 4, 16, device="cuda")
    assert backend.specialized_attention(q, q, q, softcap=1.0, dropout_p=0.1, owner=owner) is None
    assert "dropout" in owner._hndl_attention_fallback_reason
    assert backend.specialized_attention(q, q, q, softcap=1.0, mask=torch.zeros(4, 4), owner=owner) is None
    assert "bias" in owner._hndl_attention_fallback_reason


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_mask_cache_is_bounded(monkeypatch):
    owner = backend.prepare_attention(nn.Identity())
    import torch.nn.attention.flex_attention as api
    monkeypatch.setattr(api, "create_block_mask", lambda *args, **kwargs: object())
    monkeypatch.setattr(backend, "_compiled_flex", lambda q, *args, **kwargs: q)
    for length in range(1, 22):
        q = torch.randn(1, 2, length, 16, device="cuda")
        backend.specialized_attention(q, q, q, window_size=4, owner=owner)
    assert len(owner._hndl_attention_masks) == backend._MAX_MASKS


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiler_unavailability_is_cached_but_oom_is_not_swallowed(monkeypatch):
    from torch._dynamo.exc import BackendCompilerFailed
    owner = backend.prepare_attention(nn.Identity())
    q = torch.randn(1, 2, 4, 16, device="cuda")
    calls = []
    extra = {"first_useful_frame": None} if "first_useful_frame" in inspect.signature(BackendCompilerFailed).parameters else {}
    def unavailable(*args, **kwargs):
        calls.append(True)
        raise BackendCompilerFailed(unavailable, RuntimeError("compiler unavailable"), **extra)
    monkeypatch.setattr(backend, "_compiled_flex", unavailable)
    for _ in range(2):
        assert backend.specialized_attention(q, q, q, softcap=1.0, owner=owner) is None
    assert len(calls) == 1
    assert "compiler unavailable" in owner._hndl_attention_fallback_reason
    backend.clear_attention_cache(owner)
    def out_of_memory(*args, **kwargs):
        raise BackendCompilerFailed(out_of_memory, torch.OutOfMemoryError("test oom"), **extra)
    monkeypatch.setattr(backend, "_compiled_flex", out_of_memory)
    with pytest.raises(BackendCompilerFailed):
        backend.specialized_attention(q, q, q, softcap=1.0, owner=owner)
