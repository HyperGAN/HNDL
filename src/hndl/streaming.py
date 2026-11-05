"""Conservative sequence contracts and generated incremental execution."""

from __future__ import annotations

import torch

from .errors import HNDLError


_TOKENWISE = frozenset({
    "embedding", "linear", "relu", "gelu", "silu", "sigmoid", "tanh", "elu",
    "leaky_relu", "quick_gelu", "hardswish", "identity", "dropout", "drop_path",
    "layer_norm", "rms_norm", "l2_norm", "softmax", "dy_tanh", "learned_scale",
    "scale", "cast", "add", "sub", "broadcast_mul", "broadcast_add", "where",
    "swiglu", "feed_forward", "moe", "adaptive_norm", "adaln_zero", "split",
    "chunk", "concat", "sinusoidal_embed",
})


def _axes(model):
    axes = {}
    for name, entry in model.plan.inputs.items():
        if len(entry["shape"]) == 3 or (len(entry["shape"]) == 2 and entry["dtype"] in {"int32", "int64"}):
            axes[f"input:{name}"] = 1
    if not axes:
        raise HNDLError("E_UNSUPPORTED", "incremental graphs require rank-3 sequence inputs or rank-2 token IDs")
    stateful = False
    for node in model.plan.nodes:
        layer = model.nodes["n_" + node.id]
        alias = model.plan.registry.by_identity(node.op).alias
        cached = getattr(layer, "_supports_inference_cache", False)
        builtin = type(layer).__module__.startswith(("hndl.operators.", "torch.nn."))
        if not ((builtin and (cached or alias in _TOKENWISE))
                or getattr(layer, "_hndl_streaming_safe", False)):
            raise HNDLError("E_UNSUPPORTED", f"{alias} {node.id!r} has no verified incremental sequence contract")
        streaming = [port for port, ref in node.inputs.items() if ref in axes]
        if not streaming:
            if cached:
                raise HNDLError("E_UNSUPPORTED", f"cached node {node.id!r} must consume a sequence")
            continue
        if alias in {"split", "chunk", "concat", "softmax", "l2_norm"}:
            axis = getattr(layer, "dim", getattr(layer, "axis", -1))
            rank = len(node.input_shapes[streaming[0]])
            if axis % rank in {0, 1}:
                raise HNDLError("E_UNSUPPORTED", f"incremental {alias} cannot mix the batch or sequence axis")
        if cached:
            stateful = True
        input_shape = node.input_shapes[streaming[0]]
        for port, shape in node.output_shapes.items():
            if len(shape) != 3 or shape[1] != input_shape[1]:
                raise HNDLError("E_UNSUPPORTED", f"{alias} {node.id!r} must preserve the sequence axis")
            axes[f"node:{node.id}/{port}"] = 1
    if not stateful:
        raise HNDLError("E_UNSUPPORTED", "the graph has no incremental stateful operator")
    return axes


def _shape(expected, ref, axes, count):
    if ref not in axes or not isinstance(expected, tuple):
        return expected
    result = list(expected)
    result[axes[ref]] = count
    return tuple(result)


def _check(model, value, expected, dtype, port, ref, axes, count, batch):
    if ref in axes:
        port = (port[0], _shape(port[1], ref, axes, count), port[2])
    model._check(value, _shape(expected, ref, axes, count), dtype, port, batch)


def _checked(model, inputs, cache, axes, count):
    leading = inputs[0]
    batch = leading.shape[0] if isinstance(leading, torch.Tensor) and leading.ndim else None
    _, _, incoming, nodes, outgoing = model._resolve_shapes(batch)
    values = {}
    for (_, ref, expected, dtype, port), value in zip(incoming, inputs):
        _check(model, value, expected, dtype, port, ref, axes, count, batch)
        values[ref] = value
    for layer, ins, outs, ports, port_set, node_id in nodes:
        bound = []
        for ref, expected, dtype, port in ins:
            value = values[ref]
            _check(model, value, expected, dtype, port, ref, axes, count, batch)
            bound.append(value)
        try:
            result = (layer(*bound, cache=cache, node_id=node_id)
                      if getattr(layer, "_supports_inference_cache", False) else layer(*bound))
        except Exception as error:
            model._annotate_failure(node_id, error, bound, batch, after_validation=False)
            raise
        for (ref, expected, dtype, port), value in zip(outs, model._split_result(result, ports, port_set, node_id)):
            _check(model, value, expected, dtype, port, ref, axes, count, batch)
            values[ref] = value
    results = {}
    for name, ref, expected, dtype, port in outgoing:
        value = values[ref]
        _check(model, value, expected, dtype, port, ref, axes, count, batch)
        results[name] = value
    return next(iter(results.values())) if model._single_output else results


def execute(model, inputs, cache, *, single):
    """Validate a streaming signature once, then call its generated executor."""
    from . import torch as runtime
    from .inference import InferenceCache
    from .prepared import execution_module

    if not isinstance(cache, InferenceCache):
        raise HNDLError("E_RUNTIME", "cache must be an InferenceCache instance")
    if runtime._is_compiling():
        raise HNDLError("E_UNSUPPORTED", "incremental decoding is eager-only; compile the full-sequence executor separately")
    cache._begin_call(model, inputs)
    success = False
    try:
        generation = runtime._watch_generation
        token = (model._preparation_epoch, generation)
        if getattr(model, "_stream_token", None) != token or model._stores_changed():
            model._verify_state("build")
            model._stream_axes = _axes(model)
            object.__setattr__(model, "_stream_executor", execution_module(model, cached=True))
            model._stream_validated = set()
            model._stream_token = token
        axes = model._stream_axes
        counts = {value.shape[axes[f"input:{name}"]]
                  for name, value in zip(model._input_names, inputs) if f"input:{name}" in axes}
        if len(counts) != 1 or next(iter(counts)) < 1:
            raise HNDLError("E_RUNTIME", "incremental sequence inputs must have the same positive chunk length")
        count = next(iter(counts))
        signature = model._signature(inputs)
        if signature not in model._stream_validated:
            result = _checked(model, inputs, cache, axes, count)
            model._verify_state("forward")
            model._state_checked(runtime._watch_generation)
            if len(model._stream_validated) >= 64:
                model._stream_validated.clear()
            model._stream_validated.add(signature)
        else:
            result = model._stream_executor(cache, *inputs)
            if generation != runtime._watch_generation:
                model._verify_state("forward")
                model._state_checked(runtime._watch_generation)
        success = True
        if model._single_output:
            return result if single else runtime.OutputDict({next(iter(model.plan.outputs)): result})
        return runtime.OutputDict(result)
    finally:
        cache._end_call(success)
