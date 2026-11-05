"""Optional fused softcap/window attention, with bounded per-module metadata."""

from collections import OrderedDict
import importlib

import torch


_MAX_MASKS = 16
_compiled_flex = None


def prepare_attention(module, *, backend="auto"):
    """Select specialized attention for prepared models, leaving SDPA unchanged.

    ``flex`` requests the same supported path as ``auto`` and raises if a
    specialized call cannot use it. ``reference`` always retains eager math.
    """
    if backend not in ("auto", "flex", "reference"):
        raise ValueError("attention backend must be 'auto', 'flex', or 'reference'")
    for child in module.modules():
        child._hndl_flex_attention = backend != "reference"
        child._hndl_attention_requested = backend
        child._hndl_attention_masks = OrderedDict()
        child._hndl_attention_backend = "reference"
    return module


def clear_attention_cache(module):
    """Release retained block mask tensors, for example before a device move."""
    for child in module.modules():
        cache = getattr(child, "_hndl_attention_masks", None)
        if cache is not None:
            cache.clear()


def _fallback(owner, reason):
    if owner is not None:
        owner._hndl_attention_backend = "reference"
        owner._hndl_attention_fallback_reason = reason
        if getattr(owner, "_hndl_attention_requested", "auto") == "flex":
            raise RuntimeError(f"FlexAttention is unavailable: {reason}")
    return None


def specialized_attention(q, k, v, *, causal=False, window_size=0,
                          softcap=0.0, dropout_p=0.0, mask=None,
                          query_offset=0, owner=None):
    """Try fused attention; return None when the existing implementation is needed.

    Call before constructing a dense window mask. A supplied dense mask (for
    example learned relative position bias) intentionally uses the reference
    implementation. The offset aligns chunk queries with their prefix keys.
    """
    if owner is not None and not getattr(owner, "_hndl_flex_attention", False):
        return None
    if torch.compiler.is_compiling():
        return None
    if not (window_size > 0 or softcap > 0.0):
        return None
    if mask is not None:
        return _fallback(owner, "dense attention bias")
    if dropout_p != 0.0:
        return _fallback(owner, "attention dropout")
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return _fallback(owner, "device or dtype")
    if (q.shape[-1] > 128 or q.shape[-1] % 8 or q.shape[-1] != v.shape[-1]
            or q.shape[-2] == 0 or k.shape[-2] == 0):
        return _fallback(owner, "head dimensions or empty sequence")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        return _fallback(owner, "CUDA capability below SM80")
    try:
        api = importlib.import_module("torch.nn.attention.flex_attention")
    except ImportError:
        return _fallback(owner, "PyTorch has no FlexAttention")

    block_mask = None
    if causal or window_size > 0:
        def allowed(batch, head, query, key):
            delta = query + query_offset - key
            if causal:
                return (delta >= 0) & (delta < window_size) if window_size > 0 else delta >= 0
            return delta.abs() < window_size

        key = (q.shape[-2], k.shape[-2], str(q.device), bool(causal), window_size, query_offset)
        cache = None if owner is None else owner.__dict__.setdefault("_hndl_attention_masks", OrderedDict())
        if cache is not None and key in cache:
            block_mask = cache[key]
            cache.move_to_end(key)
        else:
            block_mask = api.create_block_mask(allowed, None, None, q.shape[-2], k.shape[-2],
                                               device=str(q.device), _compile=True)
            if cache is not None:
                cache[key] = block_mask
                while len(cache) > _MAX_MASKS:
                    cache.popitem(last=False)

    score_mod = None
    if softcap > 0.0:
        def cap(score, batch, head, query, key):
            return softcap * torch.tanh(score / softcap)
        score_mod = cap

    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(api.flex_attention, fullgraph=True, dynamic=False)
    try:
        result = _compiled_flex(q, k, v, score_mod=score_mod, block_mask=block_mask,
                                enable_gqa=q.shape[1] != k.shape[1])
    except NotImplementedError as error:
        return _fallback(owner, str(error))
    if owner is not None:
        owner._hndl_attention_backend = "flex"
        owner.__dict__.pop("_hndl_attention_fallback_reason", None)
    return result
