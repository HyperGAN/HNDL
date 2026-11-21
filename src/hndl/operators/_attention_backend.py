"""Optional fused softcap/window attention, with bounded per-module metadata."""

from collections import OrderedDict
from functools import lru_cache
import importlib
import inspect

import torch


_MAX_MASKS = 16
# Older compiler APIs specialize on concrete sizes. PyTorch 2.6 needs them
# when selecting backward tiles and ordering packed Q/K/V strides.
_ISOLATED_FLEX = "isolate_recompiles" in inspect.signature(torch.compile).parameters
_UNSAFE_FLEX_BACKWARD = torch.__version__.startswith("2.6.")


def _capped_score(owner, softcap, device):
    # A captured Python float becomes a symbolic scalar under dynamic=True,
    # which FlexAttention cannot lower inside a differentiable score modifier.
    cached = getattr(owner, "_hndl_attention_softcap", None)
    if cached is not None and cached[0] == softcap and cached[1].device == device:
        cap = cached[1]
    else:
        with torch.inference_mode(False):
            # Finish this one-time copy before publishing the constant to
            # other streams, including a prepared executor's capture stream.
            host = torch.tensor(softcap, device="cpu", dtype=torch.float32)
            cap = host.to(device, non_blocking=False)
        if owner is not None:
            owner._hndl_attention_softcap = (softcap, cap)
    cap.record_stream(torch.cuda.current_stream(device))

    def score_mod(score, batch, head, query, key):
        return cap * torch.tanh(score / cap)

    return score_mod


@lru_cache(maxsize=32)
def _flex_kernel(configuration):
    from torch.nn.attention.flex_attention import flex_attention

    if _ISOLATED_FLEX:
        return torch.compile(flex_attention, isolate_recompiles=True, dynamic=True, fullgraph=True)

    # Older PyTorch tracks the recompile budget by Python code object. Give
    # each configuration its own entry point while keeping the same kernels.
    def run(*args, **kwargs):
        return flex_attention(*args, **kwargs)

    run.__code__ = run.__code__.replace()
    return torch.compile(run, dynamic=False, fullgraph=True)


def _compiled_flex(q, k, v, *, score_mod, block_mask, enable_gqa, fixed_mask=()):
    # Flex kernels specialize on head layout and execution mode. Sequence
    # lengths stay dynamic within each configuration (e.g. chunked decoding).
    fixed_shapes = () if _ISOLATED_FLEX else (q.shape, k.shape, fixed_mask)
    kernel = _flex_kernel((q.device, q.dtype, q.shape[1], k.shape[1], q.shape[-1], fixed_shapes,
                           torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                           score_mod is not None, block_mask is not None))
    return kernel(q, k, v, score_mod=score_mod, block_mask=block_mask, enable_gqa=enable_gqa)


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
        child._hndl_attention_failures = OrderedDict()
        child.__dict__.pop("_hndl_attention_softcap", None)
        child._hndl_attention_backend = "reference"
    return module


def clear_attention_cache(module):
    """Release derived mask and softcap tensors, for example before a move."""
    for child in module.modules():
        child.__dict__.pop("_hndl_attention_softcap", None)
        cache = getattr(child, "_hndl_attention_masks", None)
        if cache is not None:
            cache.clear()
        failures = getattr(child, "_hndl_attention_failures", None)
        if failures is not None:
            failures.clear()


def _fallback(owner, reason):
    if owner is not None:
        owner._hndl_attention_backend = "reference"
        owner._hndl_attention_fallback_reason = reason
        if getattr(owner, "_hndl_attention_requested", "auto") == "flex":
            raise RuntimeError(f"FlexAttention is unavailable: {reason}")
    return None


def _compiler_unavailable(error):
    cause = error
    while cause is not None:
        if isinstance(cause, torch.OutOfMemoryError):
            return False
        cause = getattr(cause, "inner_exception", None) or cause.__cause__
    return (isinstance(error, (NotImplementedError, ImportError))
            or type(error).__module__.startswith(("torch._dynamo", "torch._inductor", "triton")))


def _failed(owner, failures, key, error):
    if not _compiler_unavailable(error):
        raise error
    reason = str(error)[:512]
    if failures is not None:
        failures[key] = reason
        while len(failures) > _MAX_MASKS:
            failures.popitem(last=False)
    return _fallback(owner, reason)


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
    if (q.shape[-1] < 16 or q.shape[-1] > 128 or q.shape[-1] % 8 or q.shape[-1] != v.shape[-1]
            or q.shape[-2] == 0 or k.shape[-2] == 0):
        return _fallback(owner, "head dimensions or empty sequence")
    if _UNSAFE_FLEX_BACKWARD and torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _fallback(owner, "PyTorch 2.6 FlexAttention backward can return incorrect key/value gradients")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        return _fallback(owner, "CUDA capability below SM80")
    try:
        api = importlib.import_module("torch.nn.attention.flex_attention")
    except ImportError:
        return _fallback(owner, "PyTorch has no FlexAttention")

    failure_key = (q.shape, k.shape, v.shape, q.dtype, q.device, causal, window_size,
                   softcap, query_offset, torch.is_grad_enabled(), q.requires_grad, k.requires_grad, v.requires_grad)
    failures = None if owner is None else owner.__dict__.setdefault("_hndl_attention_failures", OrderedDict())
    if failures is not None and failure_key in failures:
        return _fallback(owner, failures[failure_key])

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
            try:
                with torch.inference_mode(False), torch.no_grad():
                    block_mask = api.create_block_mask(allowed, None, None, q.shape[-2], k.shape[-2],
                                                       device=str(q.device), _compile=True)
            except Exception as error:
                return _failed(owner, failures, failure_key, error)
            if cache is not None:
                cache[key] = block_mask
                while len(cache) > _MAX_MASKS:
                    cache.popitem(last=False)

    score_mod = _capped_score(owner, softcap, q.device) if softcap > 0.0 else None

    try:
        fixed_mask = () if _ISOLATED_FLEX else (causal, window_size, query_offset)
        result = _compiled_flex(q, k, v, score_mod=score_mod, block_mask=block_mask,
                                enable_gqa=q.shape[1] != k.shape[1], fixed_mask=fixed_mask)
    except Exception as error:
        return _failed(owner, failures, failure_key, error)
    if owner is not None:
        owner._hndl_attention_backend = "flex"
        owner.__dict__.pop("_hndl_attention_fallback_reason", None)
    return result
