"""Bounded state for incremental, inference-only sequence execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
import operator
import threading
from typing import Any
import weakref

import torch
from torch.nn import functional as F

from .errors import HNDLError


_MISSING = object()
_chain = chain.from_iterable
_get_ver = operator.attrgetter("_version")
_get_ptr = torch.Tensor.data_ptr
_get_shape = operator.attrgetter("shape")
_get_dtype = operator.attrgetter("dtype")
_get_modules = operator.attrgetter("_modules")
_get_parameters = operator.attrgetter("_parameters")
_get_buffers = operator.attrgetter("_buffers")


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
    from .torch import GraphModule

    if isinstance(module, GraphModule):
        # Build and rebind already record the module traversal. Read each
        # module's *current* dictionaries so replacing a Parameter, buffer or
        # a whole registration dictionary cannot reuse an old signature.
        tensors = []
        for row in module._state_program:
            owner, children = row[0], row[3]
            if len(owner._modules) != len(children) or any(
                owner._modules.get(name, _MISSING) is not child for name, child in children
            ):
                raise HNDLError("E_RUNTIME", "cache model structure changed; rebind_state() and reset() first")
            tensors.extend(owner._parameters.values())
            tensors.extend(owner._buffers.values())
    else:
        tensors = chain(module.parameters(), module.buffers())
    return tuple((id(t), _state_version(t), t.data_ptr(), t.shape, t.stride(), t.dtype, t.device)
                 for t in tensors if t is not None)


_DEFAULT_PARAMETER_SIGNATURE = _parameter_signature


def _graph_guard_plan(model):
    """Build and cache structural views on the GraphModule itself (no extra refs in cache)."""
    state_program = model._state_program
    state_prog_id = id(state_program)
    plan = getattr(model, "_inference_guard_plan", None)
    if plan is not None and plan[0] == state_prog_id:
        return plan
    owners = tuple(row[0] for row in state_program)
    mod_dict_ids = tuple(id(o._modules) for o in owners)
    param_dict_ids = tuple(id(o._parameters) for o in owners)
    buf_dict_ids = tuple(id(o._buffers) for o in owners)
    mod_views = tuple(o._modules.values() for o in owners)
    mod_lens = tuple(len(row[3]) for row in state_program)
    mod_child_ids = tuple(id(child) for row in state_program for _, child in row[3])
    mod_key_views = tuple(o._modules.keys() for o in owners)
    mod_keys = tuple(tuple(name for name, _ in row[3]) for row in state_program)
    all_stores = tuple(s for o in owners for s in (o._parameters, o._buffers))
    empty_stores = tuple(s for s in all_stores if not s)
    empty_copies = tuple({} for _ in empty_stores)
    store_views = tuple(s.values() for s in all_stores if s)
    store_lens = tuple(len(v) for v in store_views)
    plan = (
        state_prog_id, owners, mod_dict_ids, param_dict_ids, buf_dict_ids,
        mod_views, mod_lens, mod_child_ids, mod_key_views, mod_keys,
        empty_stores, empty_copies, store_views, store_lens,
    )
    object.__setattr__(model, "_inference_guard_plan", plan)
    return plan


def _snapshot_graph_guard(model):
    """Capture pure-value tensor metadata for fast steady-state verification."""
    plan = _graph_guard_plan(model)
    (state_prog_id, owners, mod_dict_ids, param_dict_ids, buf_dict_ids,
     mod_views, mod_lens, mod_child_ids, mod_key_views, mod_keys,
     empty_stores, empty_copies, store_views, store_lens) = plan
    # Confirm live dictionaries currently match the plan before snapshotting.
    if (tuple(map(id, map(_get_modules, owners))) != mod_dict_ids
            or tuple(map(len, mod_views)) != mod_lens
            or tuple(map(id, _chain(mod_views))) != mod_child_ids
            or tuple(map(tuple, mod_key_views)) != mod_keys
            or tuple(map(id, map(_get_parameters, owners))) != param_dict_ids
            or tuple(map(id, map(_get_buffers, owners))) != buf_dict_ids
            or empty_stores != empty_copies
            or tuple(map(len, store_views)) != store_lens):
        return None
    tensors = tuple(_chain(store_views))
    all_ids = tuple(map(id, tensors))
    has_nones = any(t is None for t in tensors)
    non_none = tuple(t for t in tensors if t is not None) if has_nones else tensors
    try:
        vers = tuple(map(_get_ver, non_none))
    except RuntimeError:
        return None
    ptrs = tuple(map(_get_ptr, non_none))
    shapes = tuple(map(_get_shape, non_none))
    dtypes = tuple(map(_get_dtype, non_none))
    return (state_prog_id, all_ids, has_nones, vers, ptrs, shapes, dtypes)


def _check_graph_guard(model, guard):
    """Fast C-level verification of model structure and parameter/buffer state.

    Returns ``True`` if unchanged, ``False`` if parameter/buffer state changed,
    and raises ``HNDLError`` if module structure changed or an inference-mode
    unversioned tensor is encountered.
    """
    state_prog_id, all_ids, has_nones, vers, ptrs, shapes, dtypes = guard
    if id(model._state_program) != state_prog_id:
        return False
    plan = _graph_guard_plan(model)
    (_, owners, mod_dict_ids, param_dict_ids, buf_dict_ids,
     mod_views, mod_lens, mod_child_ids, mod_key_views, mod_keys,
     empty_stores, empty_copies, store_views, store_lens) = plan
    if (tuple(map(id, map(_get_modules, owners))) != mod_dict_ids
            or tuple(map(len, mod_views)) != mod_lens
            or tuple(map(id, _chain(mod_views))) != mod_child_ids
            or tuple(map(tuple, mod_key_views)) != mod_keys):
        raise HNDLError("E_RUNTIME", "cache model structure changed; rebind_state() and reset() first")
    if (tuple(map(id, map(_get_parameters, owners))) != param_dict_ids
            or tuple(map(id, map(_get_buffers, owners))) != buf_dict_ids
            or empty_stores != empty_copies
            or tuple(map(len, store_views)) != store_lens):
        return False
    tensors = tuple(_chain(store_views))
    if tuple(map(id, tensors)) != all_ids:
        return False
    non_none = tuple(t for t in tensors if t is not None) if has_nones else tensors
    try:
        if tuple(map(_get_ver, non_none)) != vers:
            return False
    except RuntimeError as error:
        raise HNDLError(
            "E_RUNTIME", "cached model state must track mutation versions; create parameters and buffers "
            "outside torch.inference_mode() before using an InferenceCache",
        ) from error
    if (tuple(map(_get_ptr, non_none)) != ptrs
            or tuple(map(_get_shape, non_none)) != shapes
            or tuple(map(_get_dtype, non_none)) != dtypes):
        return False
    return True


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
        self._graph_guard = None
        self._stream_verified_token = None
        self._model_epoch = None
        self._graph_nodes = None
        self._cuda_stream = None
        self._active = False
        self._active_thread = None
        self._active_device = None
        self._active_stream = None
        self._active_autocast = None
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
                self._graph_guard = None
                self._stream_verified_token = None
                self._model_epoch = None
                self._graph_nodes = None
                self._cuda_stream = None
                self._invalid = False
                self._active = False
                self._active_thread = None
                self._active_device = None
                self._active_stream = None
                self._active_autocast = None
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
            self._active_device = None
            self._active_stream = None
            self._active_autocast = None
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
        if len(values) == 1:
            x = values[0]
            if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3):
                raise HNDLError("E_RUNTIME", "cached graph inputs must be rank-2 token IDs or rank-3 sequences")
            dev = x.device
            stream = _stream_id(dev)
            if stream is not None:
                binding = (dev, stream)
                if self._cuda_stream is not None and binding != self._cuda_stream:
                    raise HNDLError("E_RUNTIME", "cache CUDA stream changed; reset() before using a different stream")
            else:
                binding = None
            ac_enabled = torch.is_autocast_enabled(dev.type)
            ac_dtype = torch.get_autocast_dtype(dev.type)
            shape_key = (x.shape[0],) if x.ndim == 2 else (x.shape[0], x.shape[2])
            signature = ((shape_key, x.dtype, dev, ac_enabled, ac_dtype, stream),)
            active_device, active_stream, active_autocast = dev, stream, (ac_enabled, ac_dtype)
        else:
            if not values or any(not isinstance(x, torch.Tensor) or x.ndim not in (2, 3) for x in values):
                raise HNDLError("E_RUNTIME", "cached graph inputs must be rank-2 token IDs or rank-3 sequences")
            streams = tuple(_stream_id(x.device) for x in values)
            binding = self._check_cuda_stream({(x.device, stream) for x, stream in zip(values, streams)
                                               if stream is not None})
            signature = tuple((tuple(x.shape[:1]) + tuple(x.shape[2:]), x.dtype, x.device,
                               torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type), stream)
                              for x, stream in zip(values, streams))
            dev0 = values[0].device
            active_device = dev0
            active_stream = streams[0]
            active_autocast = (signature[0][3], signature[0][4])
        epoch = getattr(model, "_preparation_epoch", None)
        if self._model is not None:
            if self._model() is not model or signature != self._input_signature or epoch != self._model_epoch:
                raise HNDLError("E_RUNTIME", "cache model, parameters/buffers, batch, features, dtype or device changed; reset() first")
            if self._graph_guard is not None and _parameter_signature is _DEFAULT_PARAMETER_SIGNATURE:
                if not _check_graph_guard(model, self._graph_guard):
                    raise HNDLError("E_RUNTIME", "cache model, parameters/buffers, batch, features, dtype or device changed; reset() first")
            else:
                if _parameter_signature(model) != self._model_parameters:
                    raise HNDLError("E_RUNTIME", "cache model, parameters/buffers, batch, features, dtype or device changed; reset() first")
            if self._graph_nodes is not None:
                states = self._states
                positions = set()
                for key in self._graph_nodes:
                    st = states.get(key)
                    if not isinstance(st, _NodeState):
                        raise HNDLError("E_RUNTIME", "part of the graph cache was reset; reset() the complete session before decoding")
                    positions.add(st.position)
                if len(positions) > 1:
                    raise HNDLError("E_RUNTIME", "cached graph positions differ; reset() before decoding")
        else:
            parameters = _parameter_signature(model)
            self._model = weakref.ref(model)
            self._input_signature = signature
            self._model_parameters = parameters
            self._model_epoch = epoch
            from .torch import GraphModule
            if isinstance(model, GraphModule) and _parameter_signature is _DEFAULT_PARAMETER_SIGNATURE:
                self._graph_guard = _snapshot_graph_guard(model)
            else:
                self._graph_guard = None
        if binding is not None:
            self._cuda_stream = binding
        self._active_device = active_device
        self._active_stream = active_stream
        self._active_autocast = active_autocast
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
            self._active_device = None
            self._active_stream = None
            self._active_autocast = None
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
        if self._active and x.device == self._active_device:
            stream = self._active_stream
            binding = None
            ac_enabled, ac_dtype = self._active_autocast
        else:
            stream = _stream_id(x.device)
            binding = self._check_cuda_stream({(x.device, stream)} if stream is not None else set())
            ac_enabled = torch.is_autocast_enabled(x.device.type)
            ac_dtype = torch.get_autocast_dtype(x.device.type)
        if sequence_axis == 1:
            shape_key = (x.shape[0], x.shape[2]) if x.ndim == 3 else (x.shape[0], x.shape[2], x.shape[3])
        else:
            shape_key = tuple(size for i, size in enumerate(x.shape) if i != sequence_axis)
        signature = (shape_key, x.dtype, x.device, ac_enabled, ac_dtype, stream)
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


def _append_window(bank, chunk, offset):
    """Append to a head-major ring and return the keys/values visible to this chunk."""
    capacity = bank.shape[-2]
    count = chunk.shape[-2]
    end = offset + count
    if end <= capacity:
        bank[..., offset:end, :].copy_(chunk)
        return bank[..., :end, :]

    if count == 1:
        # Attention to one token sees every ring slot. Keys and values may
        # share the same permutation because rotary positions are already applied.
        slot = offset % capacity
        bank[..., slot:slot + 1, :].copy_(chunk)
        return bank

    # Across a chunk, each query can see a different suffix of the old window.
    # Materialize the required chronological rows before overwriting any slot.
    history = min(offset, capacity - 1)
    visible = chunk
    if history:
        first = (offset - history) % capacity
        first_count = min(history, capacity - first)
        parts = [bank[..., first:first + first_count, :]]
        if first_count < history:
            parts.append(bank[..., :history - first_count, :])
        visible = torch.cat([*parts, chunk], dim=-2)

    keep = min(capacity, count)
    tail = chunk[..., count - keep:, :]
    first = (end - keep) % capacity
    first_count = min(keep, capacity - first)
    bank[..., first:first + first_count, :].copy_(tail[..., :first_count, :])
    if first_count < keep:
        bank[..., :keep - first_count, :].copy_(tail[..., first_count:, :])
    return visible


def _kv_append(cache: InferenceCache, state: _NodeState, key, value, window_size=0):
    """Append head-major keys and values, retaining at most one local window."""
    capacity = min(cache.max_seq_len, window_size) if window_size else cache.max_seq_len
    if "key" not in state.tensors:
        state.tensors["key"] = key.new_empty(*key.shape[:-2], capacity, key.shape[-1])
        state.tensors["value"] = value.new_empty(*value.shape[:-2], capacity, value.shape[-1])
    if window_size:
        return (_append_window(state.tensors["key"], key, state.position),
                _append_window(state.tensors["value"], value, state.position))
    offset = state.position
    end = offset + key.shape[-2]
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
