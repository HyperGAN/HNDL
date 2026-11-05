"""Bounded state for incremental, inference-only sequence execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
import threading
from typing import Any
import weakref

import torch
from torch.nn import functional as F

from .errors import HNDLError


@dataclass
class _NodeState:
    owner: weakref.ReferenceType
    kind: str
    signature: tuple
    parameters: tuple
    position: int = 0
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


def _state_version(tensor):
    try:
        return tensor._version
    except RuntimeError as error:
        raise HNDLError(
            "E_RUNTIME", "cached model state must track mutation versions; create parameters and buffers "
            "outside torch.inference_mode() before using an InferenceCache",
        ) from error


def _parameter_signature(module):
    # Frozen and quantized weights may be buffers rather than Parameters.
    return tuple((id(t), _state_version(t), t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device)
                 for t in chain(module.parameters(), module.buffers()))


def _stream_id(device):
    return torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None


class InferenceCache:
    """KV and recurrent state with a fixed maximum sequence length.

    Create one cache per independent decoding session. Calls append new tokens;
    ``reset()`` starts a fresh session. Batch, feature shape, dtype, device and
    owning module must stay fixed until reset. State is never part of a model's
    ``state_dict``. The mapping interface also accepts application-owned state.
    """

    def __init__(self, max_seq_len: int = 2048) -> None:
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        self._max_seq_len = max_seq_len
        self._states: dict[str, Any] = {}
        self._model = None
        self._input_signature = None
        self._model_parameters = None
        self._model_epoch = None
        self._graph_nodes = None
        self._cuda_stream = None
        self._active = False
        self._active_thread = None
        self._session_lock = threading.Lock()
        self._invalid = False

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def get(self, node_id: str, default: Any = None) -> Any:
        return self._states.get(node_id, default)

    def set(self, node_id: str, state: Any) -> None:
        self._states[node_id] = state

    def reset(self, node_id: str | None = None) -> None:
        if not self._session_lock.acquire(blocking=False):
            raise HNDLError("E_RUNTIME", "cannot reset an active InferenceCache session")
        try:
            if node_id is None:
                self._states.clear()
                self._model = None
                self._input_signature = None
                self._model_parameters = None
                self._model_epoch = None
                self._graph_nodes = None
                self._cuda_stream = None
                self._invalid = False
                self._active = False
                self._active_thread = None
            else:
                self._states.pop(node_id, None)
                # A composite operator uses hierarchical child keys.
                for key in tuple(self._states):
                    if key.startswith(node_id + "/"):
                        del self._states[key]
                if self._model is None and not any(isinstance(state, _NodeState) for state in self._states.values()):
                    self._cuda_stream = None
        finally:
            self._session_lock.release()

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._states

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, node_id: str) -> Any:
        return self._states[node_id]

    def __setitem__(self, node_id: str, state: Any) -> None:
        self._states[node_id] = state

    def __delitem__(self, node_id: str) -> None:
        del self._states[node_id]

    def keys(self):
        return self._states.keys()

    def values(self):
        return self._states.values()

    def items(self):
        return self._states.items()

    def position(self, node_id: str) -> int:
        state = self._states.get(node_id)
        return state.position if isinstance(state, _NodeState) else 0

    @property
    def allocated_bytes(self) -> int:
        """Bytes retained by built-in KV, convolution and recurrent state."""
        return sum(t.numel() * t.element_size()
                   for state in self._states.values() if isinstance(state, _NodeState)
                   for t in state.tensors.values())

    def __repr__(self) -> str:
        return f"InferenceCache({list(self._states)}, max_seq_len={self.max_seq_len})"

    def _begin_call(self, model, inputs):
        """Bind a graph session; the runtime validates its streamable time axes."""
        if not self._session_lock.acquire(blocking=False):
            raise HNDLError("E_RUNTIME", "an InferenceCache cannot be reused concurrently or recursively")
        try:
            self._active_thread = threading.get_ident()
            self._bind_call(model, inputs)
        except BaseException:
            self._active = False
            self._active_thread = None
            self._session_lock.release()
            raise

    def _bind_call(self, model, inputs):
        if self._invalid:
            raise HNDLError("E_RUNTIME", "the previous cached call failed; reset() before decoding again")
        if self._active:
            raise HNDLError("E_RUNTIME", "an InferenceCache cannot be reused concurrently or recursively")
        if model.training or torch.is_grad_enabled():
            raise HNDLError("E_RUNTIME", "incremental decoding requires eval() and torch.no_grad() or inference_mode()")
        values = tuple(inputs.values()) if isinstance(inputs, dict) else tuple(inputs)
        if not values or any(not isinstance(x, torch.Tensor) or x.ndim not in (2, 3) for x in values):
            raise HNDLError("E_RUNTIME", "cached graph inputs must be rank-2 token IDs or rank-3 sequences")
        streams = tuple(_stream_id(x.device) for x in values)
        binding = self._check_cuda_stream({(x.device, stream) for x, stream in zip(values, streams)
                                           if stream is not None})
        signature = tuple((tuple(x.shape[:1]) + tuple(x.shape[2:]), x.dtype, x.device,
                           torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type), stream)
                          for x, stream in zip(values, streams))
        parameters = _parameter_signature(model)
        epoch = getattr(model, "_preparation_epoch", None)
        if self._model is not None and (self._model() is not model or signature != self._input_signature
                                       or parameters != self._model_parameters or epoch != self._model_epoch):
            raise HNDLError("E_RUNTIME", "cache model, parameters/buffers, batch, features, dtype or device changed; reset() first")
        if self._graph_nodes is not None and any(not isinstance(self._states.get(key), _NodeState) for key in self._graph_nodes):
            raise HNDLError("E_RUNTIME", "part of the graph cache was reset; reset() the complete session before decoding")
        positions = {self._states[key].position for key in (self._graph_nodes or ())}
        if len(positions) > 1:
            raise HNDLError("E_RUNTIME", "cached graph positions differ; reset() before decoding")
        self._model = weakref.ref(model)
        self._input_signature = signature
        self._model_parameters = parameters
        self._model_epoch = epoch
        if binding is not None:
            self._cuda_stream = binding
        self._active = True

    def _check_cuda_stream(self, streams):
        if len(streams) > 1:
            raise HNDLError("E_RUNTIME", "an InferenceCache session requires one CUDA device and stream")
        binding = next(iter(streams), None)
        if binding is not None and self._cuda_stream is not None and binding != self._cuda_stream:
            raise HNDLError("E_RUNTIME", "cache CUDA stream changed; reset() before using a different stream")
        return binding

    def _end_call(self, success):
        if not self._active or self._active_thread != threading.get_ident():
            raise HNDLError("E_RUNTIME", "only the active InferenceCache session may finish its call")
        try:
            self._finish_call(success)
        except BaseException:
            self._invalid = True
            raise
        finally:
            self._active = False
            self._active_thread = None
            self._session_lock.release()

    def _finish_call(self, success):
        if not success:
            self._invalid = True
        else:
            nodes = frozenset(key for key, value in self._states.items() if isinstance(value, _NodeState))
            positions = {self._states[key].position for key in nodes}
            if len(positions) > 1:
                self._invalid = True
                raise HNDLError("E_RUNTIME", "cached graph operators consumed different positions; reset() before decoding")
            if self._graph_nodes is not None and self._graph_nodes != nodes:
                self._invalid = True
                raise HNDLError("E_RUNTIME", "cached graph operators changed; reset() before decoding")
            self._graph_nodes = nodes

    def _node(self, module, node_id: str | None, x: torch.Tensor, kind: str,
              sequence_axis: int = 1) -> _NodeState:
        if self._session_lock.locked() and self._active_thread != threading.get_ident():
            raise HNDLError("E_RUNTIME", "an InferenceCache cannot be reused concurrently")
        if self._invalid:
            raise HNDLError("E_RUNTIME", "the previous cached call failed; reset() before decoding again")
        if module.training or torch.is_grad_enabled():
            raise HNDLError("E_RUNTIME", "incremental decoding requires eval() and torch.no_grad() or inference_mode()")
        if torch.compiler.is_compiling():
            raise HNDLError("E_UNSUPPORTED", "incremental cache mutation is not supported under torch.compile")
        if not isinstance(node_id, str) or not node_id:
            raise HNDLError("E_RUNTIME", "cached operator calls require a nonempty node_id")
        if x.ndim not in (3, 4) or x.shape[0] < 1 or x.shape[sequence_axis] < 1:
            raise HNDLError("E_RUNTIME", "incremental decoding expects a nonempty sequence tensor")
        stream = _stream_id(x.device)
        binding = self._check_cuda_stream({(x.device, stream)} if stream is not None else set())
        signature = (tuple(size for i, size in enumerate(x.shape) if i != sequence_axis), x.dtype, x.device,
                     torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type), stream)
        state = self._states.get(node_id)
        # The graph runtime has just checked every parameter in _begin_call.
        # Repeating subtree walks for existing operator states adds avoidable
        # Python work to each token. Direct layer calls keep their own guards.
        parameters = (state.parameters if self._active and isinstance(state, _NodeState)
                      else _parameter_signature(module))
        if state is None:
            state = _NodeState(weakref.ref(module), kind, signature, parameters)
            self._states[node_id] = state
        elif (not isinstance(state, _NodeState) or state.owner() is not module
              or state.kind != kind or state.signature != signature
              or state.parameters != parameters):
            raise HNDLError("E_RUNTIME", f"cache state for {node_id!r} has a different owner, parameters/buffers, batch, width, dtype or device; reset it first")
        if state.position + x.shape[sequence_axis] > self.max_seq_len:
            raise HNDLError("E_RUNTIME", f"cache capacity {self.max_seq_len} exceeded at {node_id!r}; reset or create a larger cache")
        if binding is not None:
            self._cuda_stream = binding
        return state


def _causal_conv(state: _NodeState, name: str, x, weight, bias, groups):
    """Depthwise causal convolution of a new chunk, retaining K-1 raw inputs."""
    history_len = weight.shape[-1] - 1
    if history_len == 0:
        return F.conv1d(x, weight, bias, groups=groups)
    history = state.tensors.get(name)
    if history is None:
        history = torch.zeros(*x.shape[:-1], history_len, device=x.device, dtype=x.dtype)
        state.tensors[name] = history
    window = torch.cat((history, x), dim=-1)
    result = F.conv1d(window, weight, bias, groups=groups)
    history.copy_(window[..., -history_len:])
    return result


def _kv_append(cache: InferenceCache, state: _NodeState, key, value):
    """Append head-major keys and values without copying the existing prefix."""
    offset = state.position
    end = offset + key.shape[-2]
    if "key" not in state.tensors:
        state.tensors["key"] = key.new_empty(*key.shape[:-2], cache.max_seq_len, key.shape[-1])
        state.tensors["value"] = value.new_empty(*value.shape[:-2], cache.max_seq_len, value.shape[-1])
    state.tensors["key"][..., offset:end, :].copy_(key)
    state.tensors["value"][..., offset:end, :].copy_(value)
    return state.tensors["key"][..., :end, :], state.tensors["value"][..., :end, :]


def _decode_mask(offset, count, end, device, window_size=0):
    """Bottom-right causal alignment, including a causal sliding window."""
    query = torch.arange(offset, offset + count, device=device)[:, None]
    key = torch.arange(end, device=device)[None, :]
    allowed = key <= query
    if window_size > 0:
        allowed &= query - key < window_size
    return allowed
