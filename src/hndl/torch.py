"""PyTorch construction and execution for concrete HNDL plans."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import copy
import heapq
import itertools
import operator
import os
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
import weakref

import torch
from torch import nn
from torch.nn.modules import module as _torch_module

from .errors import HNDLError, describe_iterations
from .operator import COMPUTE_DTYPES
from .settings import MATRIX_SCHEMES
from .types import _is_chain, batch_multiple, contract_header

if TYPE_CHECKING:
    from .registry import Registry
    from .types import ResolvedPlan

__all__ = [
    "GraphModule",
    "InferenceCache",
    "Network",
    "build",
    "construct",
    "load_weights",
    "network",
    "network_file",
    "network_from_callable",
    "parameter_counts",
]

# Build metadata a copied network shares with its original: immutable records
# describing the resolved architecture, never the parameters that train.
SHARED_METADATA = frozenset({"plan", "build_receipt", "_port_orders", "_port_dtypes",
                             "_input_names", "_input_set", "_input_dtypes", "_output_dtypes",
                             "_build_dtype", "_node_labels", "_single_output",
                             "_spec_inputs", "_spec_outputs", "_spec_in", "_spec_out",
                             "_symbolic_compute_ports", "_output_is_compute", "checkpoint_nodes"})

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float64": torch.float64, "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}

#: Distinguishes "no batch resolved yet" from a legitimately resolved ``None``.
_UNSET = object()

#: The resolved-shape cache before any call: batch, dtype, then three programs.
_UNCOMPILED = (_UNSET, None, (), (), ())

#: How many distinct validated input signatures a graph remembers before it
#: forgets them all and starts again; bounds the cache under varying batch sizes.
_MAX_SIGNATURES = 64


class InferenceCache:
    """Stateful cache container managing key-value and recurrent decoding states."""

    def __init__(self) -> None:
        self._states: dict[str, Any] = {}

    def get(self, node_id: str, default: Any = None) -> Any:
        """Get cached state for node_id, or default if not present."""
        return self._states.get(node_id, default)

    def set(self, node_id: str, state: Any) -> None:
        """Set cached state for node_id."""
        self._states[node_id] = state

    def reset(self, node_id: str | None = None) -> None:
        """Reset and clear cached states (or a specific node_id)."""
        if node_id is None:
            self._states.clear()
        else:
            self._states.pop(node_id, None)

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

    def __repr__(self) -> str:
        return f"InferenceCache({list(self._states.keys())})"

#: How the note HNDL adds to an exception raised inside a node begins.
_NOTE_PREFIX = "HNDL:"

_is_compiling = torch.compiler.is_compiling
_compiling_simulated = False


@contextmanager
def _compiling():
    """Simulate torch.compiler.is_compiling() for testing."""
    global _is_compiling, _compiling_simulated
    saved_fn = _is_compiling
    saved_flag = _compiling_simulated

    def _always_compiling():
        return True

    _is_compiling = _always_compiling
    _compiling_simulated = True
    try:
        yield
    finally:
        _is_compiling = saved_fn
        _compiling_simulated = saved_flag
# One C call answering "is any autocast region active"; the public query needs
# a device type per call and costs three times as much.
_autocast_enabled = getattr(torch._C, "_is_any_autocast_enabled", None) or (
    lambda: torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"))

# Every module inside a built graph is watched. Registering a parameter,
# buffer or submodule on one --- ``register_*`` or plain attribute assignment,
# including assigning ``None`` over a registered buffer or submodule ---
# advances ``_watch_generation``, which is how a graph learns, for the price
# of one integer comparison per call, that its registered state may have
# changed. ``itertools.count`` hands out unique values even under concurrent
# bumps, so a generation observed once is never observed again after a change.
#
# PyTorch runs no hook for three edits: ``del`` of a registered name, assigning
# ``None`` over a registered *parameter*, and writing ``_parameters``,
# ``_buffers`` or ``_modules`` directly. Each graph therefore also records
# what those dictionaries held when its state last matched the build ---
# which names, which of them were ``None``, and which submodules --- and
# every call compares that record with the live dictionaries in C (see
# ``GraphModule._record_stores``).
#
# Keyed by ``id`` rather than held in a WeakSet so that a module defining
# ``__eq__`` without ``__hash__`` can still be watched; an entry disappears
# with its module, before that ``id`` can be reused.
_watched = weakref.WeakValueDictionary()
_ticks = itertools.count(1)
_watch_generation = 0

_Tensor = torch.Tensor
_chain = itertools.chain.from_iterable
_is = operator.is_
#: Paired with dictionary values by ``map(_is, ...)``; never exhausted.
_NONES = itertools.repeat(None)


def _registration_hook(module, name, value):
    global _watch_generation
    wr = _watched.data.get(id(module))
    if wr is not None and wr() is module:
        _watch_generation = next(_ticks)
    # Returning None leaves the registration exactly as the caller asked.


def _watch(state_program):
    for row in state_program:
        _watched[id(row[0])] = row[0]


for _register in (_torch_module.register_module_parameter_registration_hook,
                  _torch_module.register_module_buffer_registration_hook,
                  _torch_module.register_module_module_registration_hook):
    _register(_registration_hook)
del _register


def _dtype_text(dtype):
    return "any" if dtype is None else str(dtype).removeprefix("torch.")


def _contract_text(spec, batch):
    """A compiled contract shape in HNDL notation, batch entries bound: ``[B=32, 64]``."""
    parts = []
    for dimension in spec:
        if type(dimension) is tuple:
            multiple, text = dimension
            parts.append(text if multiple is None or batch is None else f"{text}={multiple * batch}")
        else:
            parts.append(str(dimension))
    return "[" + ", ".join(parts) + "]"


def _tensor_text(value):
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    return f"{_shape_text(tuple(value.shape))}:{_dtype_text(value.dtype)} on {value.device}"


def _module_kind(module):
    return "None" if module is None else type(module).__name__


def _source_position(node):
    """The ``(line, column, iterations)`` a declarative frontend recorded for a node, if any.

    ``iterations`` holds one ``(index, count)`` per config loop the node was
    unrolled from, outermost first.
    """
    source = node.source if isinstance(node.source, Mapping) else {}
    return source.get("line"), source.get("column"), source.get("iterations")


def _compile_shape(shape):
    """A contract shape with its batch entries pre-parsed, once, at build time.

    Every dimension becomes either a plain ``int`` or a
    ``(multiple_or_None, original_text)`` pair, so resolving a port against a
    concrete batch is a multiplication rather than a regular-expression match.
    """
    return tuple(d if not isinstance(d, str) else (batch_multiple(d), d) for d in shape)


class _NodeModules(nn.ModuleDict):
    """Registered modules whose graph structure is fixed after construction."""

    def __init__(self, modules):
        self._locked = False
        super().__init__(modules)
        self._locked = True

    def add_module(self, name, module):
        if self._locked:
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().add_module(name, module)

    def __delitem__(self, key):
        raise TypeError("HNDL architecture is fixed; resolve and build a new plan")

    def clear(self):
        raise TypeError("HNDL architecture is fixed; resolve and build a new plan")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name in self._modules:
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._modules:
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__delattr__(name)


def _shape_text(shape: Sequence[Any]) -> str:
    return "[" + ", ".join(map(str, shape)) + "]"




class OutputDict(dict):
    """A mapping of model outputs that forwards tensor attributes when single-output."""

    def __getattr__(self, name: str):
        if len(self) == 1:
            val = next(iter(self.values()))
            if hasattr(val, name):
                return getattr(val, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class GraphModule(nn.Module):
    """Execute a frozen graph once per node and return named output tensors.

    Validation happens once per *input signature*, not once per call. The
    first call whose external inputs have a given shape, dtype and device ---
    in a given training mode and autocast state --- checks every port of every
    node against the plan and then checks that no module created or removed
    registered state. Later calls with the same signature compare that
    signature and then run the modules back to back.

    What was validated is forgotten when it may no longer hold: moving or
    casting the module (``.to()``, ``.cuda()``, ``.double()``, ...),
    registering a parameter, buffer or submodule on any module in the graph,
    and, on any such module, removing an entry of ``_parameters``,
    ``_buffers`` or ``_modules``, setting one to ``None``, adding one, or
    replacing a submodule (``del``, ``module.weight = None``, a direct write)
    all re-check the registered state before the next call and revalidate
    every port on it. A tensor swapped in under a registered name without a
    hook, as ``torch.func.functional_call`` does, is not a change: it runs on
    the unchecked path, like a parameter updated in place. Not detected
    either: replacing a whole ``_parameters``/``_buffers``/``_modules``
    dictionary, reordering one, removing one name and adding another in the
    same ``_parameters`` or ``_buffers`` between two calls, and swapping in a
    submodule whose ``__eq__`` says it equals the old one.

    An exception raised inside a node's module propagates as itself --- same
    type, message and traceback --- with one note naming the node, its
    operation and source line, the tensors it received, and any change to
    registered state that explains it. ``E_RUNTIME`` is kept for what HNDL's
    own checks find.
    """

    def __init__(self, plan, modules, device, receipt, port_orders, checkpoint_nodes=None):
        super().__init__()
        self.plan = plan
        self.nodes = _NodeModules(modules)
        self._runtime_device = device
        self.checkpoint_nodes = frozenset(checkpoint_nodes or ())
        # The compute dtype the plan was built at, and the one it runs at now:
        # casting the module with .double(), .half() or .to(dtype=...) moves the
        # second without rewriting the build-time declarations below.
        self._build_dtype = DTYPES[plan.dtype]
        self._runtime_dtype = self._build_dtype
        self._chain = _is_chain(plan)
        self._port_orders = port_orders
        self._input_names = tuple(plan.inputs)
        self._input_set = frozenset(self._input_names)
        self._input_dtypes = MappingProxyType(
            {name: DTYPES[entry["dtype"]] for name, entry in plan.inputs.items()})
        self._port_dtypes = {}
        self._symbolic_compute_ports = set()
        produced = {f"input:{name}": entry["dtype"] for name, entry in plan.inputs.items()}
        produced_is_compute = {f"input:{name}": entry["dtype"] in COMPUTE_DTYPES for name, entry in plan.inputs.items()}
        for node in plan.nodes:
            spec = plan.registry.by_identity(node.op)
            declared = {}
            for port in (*node.inputs, *node.outputs):
                if spec._variadic_member(spec.variadic, port):
                    raw_d = spec.inputs[0].dtype
                elif spec._variadic_member(spec.variadic_output, port):
                    raw_d = spec.outputs[0].dtype
                else:
                    raw_d = spec._port_dtypes.get(port, "compute")

                if spec.identity == "cast" and port in spec.output_ports:
                    target_dtype = node.args.get("dtype", "compute")
                    declared[port] = plan.dtype if target_dtype == "compute" else target_dtype
                    if target_dtype == "compute":
                        self._symbolic_compute_ports.add((node.id, port))
                elif port in node.inputs and raw_d == "any":
                    producer_ref = node.inputs[port]
                    if produced_is_compute.get(producer_ref, False):
                        self._symbolic_compute_ports.add((node.id, port))
                    declared[port] = spec.port_dtype(port, plan.dtype)
                else:
                    d = spec.port_dtype(port, plan.dtype)
                    declared[port] = d
                    if raw_d == "compute":
                        self._symbolic_compute_ports.add((node.id, port))
            self._port_dtypes[node.id] = {port: None if name == "any" else DTYPES[name] for port, name in declared.items()}
            for port in node.outputs:
                ref = f"node:{node.id}/{port}"
                produced[ref] = plan.dtype if declared[port] == "any" else declared[port]
                produced_is_compute[ref] = (node.id, port) in self._symbolic_compute_ports

        self._output_dtypes = MappingProxyType(
            {name: DTYPES[produced[entry["ref"]]] for name, entry in plan.outputs.items()})
        self._output_is_compute = {name: produced_is_compute.get(entry["ref"], False) for name, entry in plan.outputs.items()}
        # What an error needs to name a node: its operation and source position,
        # including the loop iterations a config unrolled it from.
        self._node_labels = MappingProxyType(
            {node.id: (self._alias(node.op), *_source_position(node)) for node in plan.nodes})
        self._state_program = self._state_snapshot()
        # Contract shapes with their batch entries pre-parsed; _resolve_shapes
        # turns these into concrete expectations once per distinct batch size.
        self._spec_inputs = {name: _compile_shape(entry["shape"]) for name, entry in plan.inputs.items()}
        self._spec_outputs = {name: _compile_shape(entry["shape"]) for name, entry in plan.outputs.items()}
        self._spec_in = {node.id: {port: _compile_shape(shape) for port, shape in node.input_shapes.items()}
                         for node in plan.nodes}
        self._spec_out = {node.id: {port: _compile_shape(shape) for port, shape in node.output_shapes.items()}
                          for node in plan.nodes}
        self._compiled = _UNCOMPILED
        self._fast = self._build_fast_program()
        # Input signatures whose calls passed every check, and the watch
        # generation and store contents at which the registered state last
        # matched the build (``None`` forces a full re-check before the next call).
        self._validated = {}
        _watch(self._state_program)
        self._state_seen = _watch_generation
        self._stores = self._record_stores()
        self._single_output = len(plan.outputs) == 1
        self.build_receipt = MappingProxyType(receipt)

    @contextmanager
    def allow_state_mutation(self):
        """Bless external parameter / submodule mutation during PEFT, quantization or adaptation."""
        self._mutation_allowed = True
        try:
            yield self
        finally:
            self.rebind_state()
            self._mutation_allowed = False

    def rebind_state(self):
        """Re-synchronize state snapshot and C-speed stores record after external modifications."""
        self._state_program = self._state_snapshot()
        _watch(self._state_program)
        global _watch_generation
        self._state_seen = _watch_generation
        self._stores = self._record_stores()
        self._fast = self._build_fast_program()
        self._validated.clear()

    def __setattr__(self, name, value):
        if name in ("nodes", "plan") and hasattr(self, name):
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ("nodes", "plan"):
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__delattr__(name)

    def __deepcopy__(self, memo):
        """Copy every module, parameter and buffer; share the resolved plan.

        ``nn.Module`` has no copy protocol of its own, so the default one
        pickles the instance dictionary, which the plan's frozen mappings
        refuse. Copying the dictionary entry by entry instead preserves the
        module internals exactly --- parameters with their ``requires_grad``
        flags, buffers, submodules, hooks and the training mode --- while the
        architecture records are shared rather than duplicated.
        """
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name == "_stores":
                continue  # rebuilt below from the copy's own modules
            # The architecture lock guards attribute writes; this rebuilds the
            # instance dictionary directly, exactly as unpickling would.
            object.__setattr__(result, name, value if name in SHARED_METADATA
                               else copy.deepcopy(value, memo))
        # Cheap insurance: the clone rebuilds its baked programs against its own
        # modules rather than trusting a structure copied mid-flight, validates
        # its first call afresh, and watches its own modules for registrations.
        object.__setattr__(result, "_compiled", _UNCOMPILED)
        object.__setattr__(result, "_fast", result._build_fast_program())
        object.__setattr__(result, "_validated", {})
        object.__setattr__(result, "_stores", result._record_stores())
        _watch(result._state_program)
        return result

    def __copy__(self):
        """A second handle on the same graph: every tensor stays shared."""
        result = type(self).__new__(type(self))
        result.__dict__.update(self.__dict__)
        return result

    def __reduce__(self):
        raise TypeError(
            f"{type(self).__name__} cannot be pickled because its plan holds frozen mappings; "
            "save plan.to_json() next to torch.save(model.state_dict()) and rebuild with "
            "hndl.torch.build(), or use copy.deepcopy() for an in-memory copy")

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # An empty tensor also tracks device moves and dtype casts for
        # parameterless graphs, which hold no state of their own to follow.
        probe = fn(torch.empty(0, device=self._runtime_device, dtype=self._runtime_dtype))
        self._runtime_device = probe.device
        # ``.double()``, ``.half()`` and ``.to(dtype=...)`` cast floating state
        # only; a probe that came back integral means no compute-dtype change.
        if probe.is_floating_point() or probe.is_complex():
            self._runtime_dtype = probe.dtype
        # Every validated signature named the old device and dtype. A move is
        # also a natural point to re-check registered state in full, which
        # catches the edits no registration hook reports (see _state_seen).
        self._validated.clear()
        self._validated = {}
        self._state_seen = None
        return self

    def _effective_dtype(self, dtype, is_compute=True):
        """The dtype a port carries now, after any floating-point cast.

        Ports declared at the plan's compute dtype follow the module through
        ``.double()``, ``.half()`` and ``.to(dtype=...)``, exactly as its
        parameters do. Integer and boolean ports --- token ids, masks --- and
        ports declared ``any`` are untouched, because those casts leave
        non-floating state alone.
        """
        if not is_compute:
            return dtype
        return self._runtime_dtype if dtype is not None and dtype == self._build_dtype else dtype

    # -- errors -------------------------------------------------------------

    def _node_name(self, node_id):
        """``linear 'head'``: the operation and the node, as an error names them."""
        return f"{self._node_labels[node_id][0]} {node_id!r}"

    def _error(self, message, node_id=None, code="E_RUNTIME"):
        if node_id is None:
            return HNDLError(code, message)
        _, line, column, iterations = self._node_labels[node_id]
        return HNDLError(code, message, node=node_id, line=line, column=column, iterations=iterations)

    def _port_error(self, port, value, dtype, batch, problem):
        """An ``E_RUNTIME`` for one port, with the contract beside the tensor."""
        where, spec, node_id = port
        lines = [f"{where}: {problem}"]
        if batch is not None and batch <= 0:
            batch = None  # an empty batch binds no B worth printing
        if isinstance(value, torch.Tensor):
            lines.append(f"  expected {_contract_text(spec, batch)}:{_dtype_text(dtype)} on {self._runtime_device}")
            lines.append(f"  got      {_tensor_text(value)}")
            first = self._input_names[0]
            if (batch is not None and where != f"input {first!r}" and value.ndim == len(spec)
                    and any(type(d) is tuple and d[0] is not None and size != d[0] * batch
                            for d, size in zip(spec, value.shape))):
                lines.append(f"  B={batch} is this call's batch size, read from input {first!r}")
        return self._error("\n".join(lines), node_id)

    def _annotate_failure(self, node_id, error, bound, batch, after_validation):
        """Add HNDL's context to an exception out of one node, as a note.

        The exception itself propagates unchanged --- same type, same message,
        same traceback --- so callers that catch ``RuntimeError`` (or an
        out-of-memory error, to shrink the batch) keep working; the note is
        what the traceback prints after the message. Everything here runs only
        after a failure, and an exception passing out through several graphs
        (a graph used as a node of another) keeps the innermost node's note.
        """
        if isinstance(error, HNDLError) and error.node is not None:
            return  # an HNDL check already named its node
        if any(isinstance(note, str) and note.startswith(_NOTE_PREFIX)
               for note in getattr(error, "__notes__", ())):
            return
        try:
            error.add_note(self._failure_note(node_id, error, bound, batch, after_validation))
        except Exception:  # noqa: BLE001 - a note must never replace the real error
            pass

    def _failure_note(self, node_id, error, bound, batch, after_validation):
        label, line, column, iterations = self._node_labels[node_id]
        position = "".join(f", {key} {value}" for key, value in (("line", line), ("column", column))
                           if value is not None)
        if iterations:
            position += f", {describe_iterations(iterations)}"
        lines = [f"{_NOTE_PREFIX} raised inside node {node_id!r} ({label}{position})"]
        if not self._state_matches():
            # State removed between calls breaks forward rather than reporting
            # itself; say so, because that is the cause the error hides.
            _, listed = self._state_summary()
            lines.append(f"  registered state no longer matches the build: {listed}")
        elif after_validation:
            # The signature was validated earlier, so a node input that now
            # breaks its contract means an upstream module changed its output.
            compiled = self._compiled
            if batch != compiled[0] or self._runtime_dtype is not compiled[1]:
                compiled = self._resolve_shapes(batch)
            entry = next(entry for entry in compiled[3] if entry[5] == node_id)
            for (_, expected, dtype, port), value in zip(entry[1], bound):
                try:
                    self._check(value, expected, dtype, port, batch)
                except HNDLError as mismatch:
                    lines.extend("  " + text for text in mismatch.message.splitlines())
                    lines.append("  (an earlier call with the same input signature passed every check, "
                                 "so an upstream module now produces a different tensor)")
                    break
        effective = self._effective_dtype
        for port, value in zip(self._port_orders[node_id], bound):
            spec = self._spec_in[node_id][port]
            dtype = effective(self._port_dtypes[node_id][port], (node_id, port) in self._symbolic_compute_ports)
            lines.append(f"  input {port!r}: got {_tensor_text(value)}; contract "
                         f"{_contract_text(spec, batch)}:{_dtype_text(dtype)} on {self._runtime_device}")
        if type(error).__module__.startswith(("torch._dynamo", "torch._inductor")):
            lines.append("  the error came from torch.compile compiling or running this node's module; "
                         "run it without torch.compile to see the eager error")
        return "\n".join(lines)

    # -- registered state ---------------------------------------------------

    def _state_snapshot(self):
        """Record what ``named_parameters``/``named_buffers`` see, per module.

        The dotted names those walks produce are a pure function of three
        things: the order ``modules()`` visits, each module's path from the
        root, and the keys each module contributes. Freezing the visited
        modules in one flat tuple --- alongside the child mapping that fixes
        every path --- lets :meth:`_state_matches` re-derive the same answer
        without recursing through ``named_modules`` or building a single string.

        Each row is ``(module, parameter_keys, buffer_keys, children, path,
        node_id)``. ``children`` is the module's ``_modules`` mapping as
        key/value pairs, so a submodule swapped out under an unchanged key is
        still a change. The key tuples are filtered exactly as PyTorch filters
        them: ``None`` slots are skipped, and a tensor already seen earlier in
        the walk is skipped, with parameters and buffers de-duplicated
        independently. ``path`` and ``node_id`` are only read to word errors.
        """
        owners = {}
        for key, layer in self.nodes._modules.items():
            for module in layer.modules():
                owners.setdefault(id(module), key.removeprefix("n_"))
        program, seen_parameters, seen_buffers = [], set(), set()
        for path, module in self.named_modules():
            keys = []
            versions = {}
            for store, seen in ((module._parameters, seen_parameters),
                                (module._buffers, seen_buffers)):
                contributed = []
                for key, value in store.items():
                    if value is None or id(value) in seen:
                        continue
                    seen.add(id(value))
                    contributed.append(key)
                    if getattr(value, "_version", None) is not None:
                        versions[key] = value._version
                keys.append(tuple(contributed))
            program.append((module, keys[0], keys[1], tuple(module._modules.items()),
                            path, owners.get(id(module)), versions))
        return tuple(program)

    def _state_matches(self):
        """Whether every module still registers exactly what it did at build.

        Replays the build-time walk against the flat program instead of
        recursing through the module tree again. Pinning every module's child
        mapping keeps the two trees identical --- nothing can be grafted in
        unseen --- so comparing each module's contributed keys answers exactly
        what comparing the dotted name tuples used to answer. ``set.add``
        returns None, so its clause always passes and only records the tensor.
        """
        seen_parameters, seen_buffers = set(), set()
        for module, parameters, buffers, children, _, _, versions in self._state_program:
            if (tuple(module._modules.items()) != children
                    or tuple([key for key, value in module._parameters.items()
                              if value is not None and id(value) not in seen_parameters
                              and not seen_parameters.add(id(value))]) != parameters
                    or tuple([key for key, value in module._buffers.items()
                              if value is not None and id(value) not in seen_buffers
                              and not seen_buffers.add(id(value))]) != buffers):
                return False
            if _compiling_simulated and any(
                getattr(module._parameters.get(k), "_version", None) != v
                for k, v in versions.items() if k in module._parameters
            ):
                return False
        return True

    def _record_stores(self):
        """What each call compares to catch the registered-state edits no hook reports.

        Returns ``(same, copies, views, shape)``, taken whenever the
        registered state has just been found to match the build. ``same``
        holds every module's ``_modules`` and every ``_parameters`` or
        ``_buffers`` that is empty, and ``copies`` a copy of each, so
        ``same == copies`` catches a submodule added, removed, set to ``None``
        or replaced, and any entry added to an empty dictionary, in C at about
        10-15 ns per dictionary.

        The non-empty ``_parameters`` and ``_buffers`` are compared by
        ``shape`` instead: the length of each (``views`` holds their values)
        followed by whether each value is ``None``. That catches an entry
        removed, added or set to ``None`` without ever touching a tensor.
        Copies of these dictionaries would hold their tensors alive, and
        comparing them would run an elementwise ``==`` whenever a tensor had
        been swapped in under the same name, as ``torch.func.functional_call``
        does on every call. A swapped tensor is deliberately not a change: it
        registers no new name and runs on the unchecked path, as a parameter
        updated in place does. Neither is a name removed and another added in
        the same dictionary, a reordered dictionary, or a whole dictionary
        replaced (``module._parameters = {...}``): the record watches the
        dictionaries themselves.
        """
        modules = [row[0] for row in self._state_program]
        tensor_stores = [store for module in modules for store in (module._parameters, module._buffers)]
        same = tuple([module._modules for module in modules] + [store for store in tensor_stores if not store])
        views = tuple([store.values() for store in tensor_stores if store])
        store_lens = tuple(map(len, views))
        tensor_ids = tuple(map(id, _chain(views)))
        shape = [*store_lens, *map(_is, _chain(views), _NONES),
                 *[(t.dtype, t.device) if t is not None else None for t in _chain(views)]]
        return same, tuple(map(dict, same)), views, store_lens, tensor_ids, shape

    def _state_checked(self, generation):
        """Record that the registered state matched the build at ``generation``."""
        self._state_seen = generation
        self._stores = self._record_stores()

    def _state_changes(self):
        """What differs from the build-time walk, as ``(node_id, text)`` pairs."""
        changes = []
        seen = {"parameter": set(), "buffer": set()}
        for module, parameters, buffers, children, path, node_id, versions in self._state_program:
            prefix = f"{path}." if path else ""
            if _compiling_simulated:
                for k, v in versions.items():
                    val = module._parameters.get(k)
                    if val is not None and getattr(val, "_version", None) != v:
                        changes.append((node_id, f"parameter '{prefix}{k}' was modified"))
            for kind, store, recorded in (("parameter", module._parameters, parameters),
                                          ("buffer", module._buffers, buffers)):
                ids, current = seen[kind], []
                for key, value in store.items():
                    if value is not None and id(value) not in ids:
                        ids.add(id(value))
                        current.append(key)
                if tuple(current) == recorded:
                    continue
                for key in recorded:
                    if key not in current:
                        how = ("removed" if key not in store else "set to None" if store[key] is None
                               else "re-registered as an alias of an earlier tensor")
                        changes.append((node_id, f"{kind} '{prefix}{key}' was {how}"))
                changes.extend((node_id, f"{kind} '{prefix}{key}' was added")
                               for key in current if key not in recorded)
                if set(current) == set(recorded):
                    changes.append((node_id, f"the {kind}s of '{path}' were re-registered in a different order"))
            now, before = module._modules, dict(children)
            if tuple(now.items()) == children:
                continue
            for key, child in before.items():
                if key not in now:
                    changes.append((node_id, f"submodule '{prefix}{key}' was removed"))
                elif now[key] is not child:
                    how = (f"filled in with {_module_kind(now[key])} (it was None at build)" if child is None
                           else f"replaced ({_module_kind(child)} -> {_module_kind(now[key])})")
                    changes.append((node_id, f"submodule '{prefix}{key}' was {how}"))
            changes.extend((node_id, f"submodule '{prefix}{key}' was added ({_module_kind(child)})")
                           for key, child in now.items() if key not in before)
            if set(now) == set(before) and all(now[key] is child for key, child in before.items()):
                changes.append((node_id, f"the submodules of '{path}' were re-registered in a different order"))
        return changes

    def _state_summary(self):
        """The first node whose registered state changed, and every change, as text."""
        changes = self._state_changes() or [(None, "the module tree differs from the one recorded at build")]
        node_id = next((owner for owner, _ in changes if owner is not None), None)
        listed = "; ".join(text for _, text in changes[:6])
        if len(changes) > 6:
            listed += f"; and {len(changes) - 6} more"
        return node_id, listed

    def _state_error(self, when):
        """``E_RUNTIME`` naming the node whose registered state changed, and how."""
        node_id, listed = self._state_summary()
        subject = self._node_name(node_id) if node_id is not None else "the network"
        phrase = "changed during forward" if when == "forward" else "no longer matches the build"
        lines = [f"registered state of {subject} {phrase}: {listed}"]
        lines.append("  HNDL fixes every parameter, buffer and submodule when it builds a plan, so state "
                     "added or removed later is not trained, seeded or saved as the plan describes; "
                     "create it in __init__, or resolve and build a new plan to change the architecture")
        return self._error("\n".join(lines), node_id)

    def _verify_state(self, when):
        if not self._state_matches():
            raise self._state_error(when)

    # -- the checked program ------------------------------------------------

    @staticmethod
    def _resolve(spec, batch):
        """A compiled shape as concrete sizes, or the entry that is not a batch axis.

        A port a batch-axis join or split produced carries ``"k*B"``, which
        means ``k`` times this call's batch, not the batch itself. An entry that
        is symbolic but not a batch dimension comes back as the original string,
        which :meth:`_check` reports from the port that actually carries it ---
        keeping that a runtime error rather than promoting it to build time.
        """
        resolved = []
        for dimension in spec:
            if type(dimension) is tuple:
                multiple, text = dimension
                if multiple is None:
                    return text
                resolved.append(None if batch is None else multiple * batch)
            else:
                resolved.append(dimension)
        return tuple(resolved)

    def _resolve_shapes(self, batch, cache=True):
        """Bind every port's expected shape and dtype to this batch, once.

        The whole program is published as a single tuple, so a caller on
        another thread reads either the entire previous program or the entire
        new one --- never a half-rebuilt mixture of the two.
        """
        resolve, effective = self._resolve, self._effective_dtype
        expected_in = {node: {p: resolve(s, batch) for p, s in ports.items()}
                       for node, ports in self._spec_in.items()}
        expected_out = {node: {p: resolve(s, batch) for p, s in ports.items()}
                        for node, ports in self._spec_out.items()}
        effdt = {node: {p: effective(d, (node, p) in self._symbolic_compute_ports) for p, d in ports.items()}
                 for node, ports in self._port_dtypes.items()}
        compiled = (
            batch, self._runtime_dtype,
            tuple((name, f"input:{name}", resolve(self._spec_inputs[name], batch),
                   effective(self._input_dtypes[name], self.plan.inputs[name]["dtype"] in COMPUTE_DTYPES),
                   (f"input {name!r}", self._spec_inputs[name], None))
                  for name in self._input_names),
            self._build_program(expected_in, expected_out, effdt),
            tuple((name, entry["ref"], resolve(self._spec_outputs[name], batch),
                   effective(self._output_dtypes[name], self._output_is_compute.get(name, False)),
                   (f"output {name!r}", self._spec_outputs[name], None))
                  for name, entry in self.plan.outputs.items()))
        if cache:
            self._compiled = compiled
        return compiled

    def _build_program(self, expected_in, expected_out, effdt):
        """Bake the per-call constants --- labels, module handles, port sets --- into tuples."""
        program = []
        for node in self.plan.nodes:
            name = self._node_name(node.id)
            program.append((
                self.nodes[f"n_{node.id}"],
                tuple((node.inputs[port], expected_in[node.id][port], effdt[node.id][port],
                       (f"{name} input {port!r} (from {node.inputs[port]})",
                        self._spec_in[node.id][port], node.id))
                      for port in self._port_orders[node.id]),
                tuple((f"node:{node.id}/{port}", expected_out[node.id][port], effdt[node.id][port],
                       (f"{name} output {port!r}", self._spec_out[node.id][port], node.id))
                      for port in node.outputs),
                node.outputs, frozenset(node.outputs), node.id))
        return tuple(program)

    def _build_fast_program(self):
        """The unchecked loop: every tensor lives in a numbered slot.

        External inputs take the first slots in declaration order and every
        node output the next free one, recycling intermediate slots when their
        last consumer finishes.
        """
        output_refs = {entry["ref"] for entry in self.plan.outputs.values()}
        last_consumer = {}
        for i, node in enumerate(self.plan.nodes):
            for in_ref in node.inputs.values():
                last_consumer[in_ref] = i
        for ref in output_refs:
            last_consumer[ref] = len(self.plan.nodes)

        slots = {f"input:{name}": index for index, name in enumerate(self._input_names)}
        num_inputs = len(self._input_names)
        free_slots = []
        free_set = set()
        next_slot = num_inputs

        program = []
        for i, node in enumerate(self.plan.nodes):
            ins = tuple(slots[node.inputs[port]] for port in self._port_orders[node.id])
            outs = []
            for port in node.outputs:
                ref = f"node:{node.id}/{port}"
                if free_slots:
                    slot = heapq.heappop(free_slots)
                    free_set.remove(slot)
                else:
                    slot = next_slot
                    next_slot += 1
                slots[ref] = slot
                outs.append(slot)

            to_clear = []
            # After outputs are allocated, recycle dead inputs
            for in_ref in node.inputs.values():
                if not in_ref.startswith("input:") and last_consumer.get(in_ref) == i:
                    dead_slot = slots[in_ref]
                    if dead_slot not in free_set:
                        heapq.heappush(free_slots, dead_slot)
                        free_set.add(dead_slot)
                        to_clear.append(dead_slot)

            # Also recycle any outputs that have no consumers and are not graph outputs
            for port in node.outputs:
                ref = f"node:{node.id}/{port}"
                if ref not in last_consumer:
                    dead_slot = slots[ref]
                    if dead_slot not in free_set:
                        heapq.heappush(free_slots, dead_slot)
                        free_set.add(dead_slot)
                        to_clear.append(dead_slot)

            fast_ins = ins[0] if len(ins) == 1 else operator.itemgetter(*ins) if len(ins) > 1 else None
            program.append((
                self.nodes[f"n_{node.id}"],
                fast_ins,
                outs[0] if len(outs) == 1 else None,
                (ins, tuple(outs), node.outputs, frozenset(node.outputs), node.id),
                tuple(to_clear),
            ))

        max_slot = max(slots.values()) if slots else num_inputs - 1
        total_slots = max(max_slot + 1, num_inputs)
        padding = (None,) * (total_slots - num_inputs)
        outputs = tuple((name, slots[entry["ref"]]) for name, entry in self.plan.outputs.items())
        return padding, tuple(program), outputs

    def _check(self, value, expected, dtype, port, batch):
        if not isinstance(value, torch.Tensor):
            raise self._port_error(port, value, dtype, batch, f"expected a tensor, got {type(value).__name__}")
        if type(expected) is str:
            raise self._port_error(port, value, dtype, batch,
                                   f"contract entry {expected!r} is not a batch dimension")
        if value.shape != expected or value.ndim == 0 or value.shape[0] <= 0:
            if (getattr(self, "_active_cache", None) is not None
                    and value.ndim == len(expected)
                    and value.ndim >= 2
                    and value.shape[0] > 0
                    and value.shape[-1] == expected[-1]):
                pass
            else:
                if value.ndim == 0:
                    problem = f"expected shape {_contract_text(port[1], batch)}, got a 0-dimensional tensor"
                elif value.shape[0] <= 0 and value.ndim == len(expected):
                    problem = f"batch size must be positive, got shape {_shape_text(tuple(value.shape))}"
                else:
                    problem = (f"expected shape {_contract_text(port[1], batch)}, "
                               f"got {_shape_text(tuple(value.shape))}")
                raise self._port_error(port, value, dtype, batch, problem)
        if dtype is not None and value.dtype != dtype:
            raise self._port_error(port, value, dtype, batch,
                                   f"expected dtype {_dtype_text(dtype)}, got {_dtype_text(value.dtype)}")
        if value.device != self._runtime_device:
            raise self._port_error(port, value, dtype, batch,
                                   f"expected device {self._runtime_device}, got {value.device}")

    def _split_result(self, result, out_ports, out_set, node_id):
        """A module's return value as one value per declared output port."""
        if isinstance(result, dict) and set(result) == out_set:
            return tuple(result[port] for port in out_ports)
        if isinstance(result, (tuple, list)) and len(result) == len(out_ports):
            return tuple(result)
        if len(out_ports) == 1:
            return (result,)
        if isinstance(result, dict):
            got = f"a dict with keys {tuple(result)}"
        elif isinstance(result, (tuple, list)):
            got = f"a {type(result).__name__} of {len(result)} values"
        else:
            got = f"one {type(result).__name__}"
        raise self._error(f"{self._node_name(node_id)} returned {got}; expected output ports {out_ports}, "
                          "as a tuple or list in that order or a dict with exactly those keys", node_id)

    def _run_checked(self, inputs, compiling, cache=None):
        """One call with every port checked before and after its node."""
        if type(inputs) is not tuple:
            inputs = tuple(inputs[n] for n in self._input_names)
        leading = inputs[0]
        batch = leading.shape[0] if isinstance(leading, torch.Tensor) and leading.ndim else None
        # The resolved expectations depend on the batch and on the dtype casts
        # ``_apply`` records; the device is read live by ``_check`` instead, so
        # ``.to(device)`` needs no invalidation.
        compiled = self._compiled
        if batch != compiled[0] or self._runtime_dtype is not compiled[1]:
            compiled = self._resolve_shapes(batch, cache=not compiling)
        _, _, input_program, node_program, output_program = compiled
        check = self._check
        values = {}
        for (name, key, expected, dtype, port), value in zip(input_program, inputs):
            check(value, expected, dtype, port, batch)
            values[key] = value
        for module, ins, outs, out_ports, out_set, node_id in node_program:
            bound = []
            for ref, expected, dtype, port in ins:
                value = values[ref]
                check(value, expected, dtype, port, batch)
                bound.append(value)
            if compiling:
                result = module(*bound)
            elif cache is not None and getattr(module, "_supports_inference_cache", False):
                result = module(*bound, cache=cache, node_id=node_id)
            elif self.training and node_id in self.checkpoint_nodes:
                if any(isinstance(a, torch.Tensor) and a.requires_grad for a in bound):
                    result = torch.utils.checkpoint.checkpoint(module, *bound, use_reentrant=False)
                else:
                    result = module(*bound)
            else:
                try:
                    result = module(*bound)
                except Exception as error:
                    self._annotate_failure(node_id, error, bound, batch, after_validation=False)
                    raise
            for (key, expected, dtype, port), value in zip(
                    outs, self._split_result(result, out_ports, out_set, node_id)):
                check(value, expected, dtype, port, batch)
                values[key] = value
        outputs = OutputDict()
        for name, ref, expected, dtype, port in output_program:
            value = values[ref]
            check(value, expected, dtype, port, batch)
            outputs[name] = value
        return outputs

    def _signature(self, inputs):
        """What a validated call is keyed on, or ``None`` for a non-tensor input."""
        if type(inputs) is not tuple:
            inputs = tuple(inputs[n] for n in self._input_names)
        if len(inputs) == 1:
            v = inputs[0]
            if not isinstance(v, torch.Tensor):
                return None
            return (self.training, _autocast_enabled(), v.shape, v.dtype, v.device)
        key = [self.training, _autocast_enabled()]
        for v in inputs:
            if not isinstance(v, torch.Tensor):
                return None
            key += (v.shape, v.dtype, v.device)
        return tuple(key)

    def _validate(self, inputs, signature, generation, stores_changed=False, cache=None):
        """The first call for a signature: check everything, then remember it."""
        if stores_changed or generation != self._state_seen:
            # A parameter, buffer or submodule was registered, removed or
            # replaced on a module of this graph since the last check: confirm
            # the state still matches the build, and forget every validated
            # signature, because a parameter replaced under the same name can
            # change any shape. The stores are recorded once, after the call.
            self._verify_state("build")
            self._validated = {}
            self._state_seen = generation
        outputs = self._run_checked(inputs, compiling=False, cache=cache)
        generation = _watch_generation
        self._verify_state("forward")
        self._state_checked(generation)
        validated = self._validated
        if len(validated) >= _MAX_SIGNATURES:
            validated.clear()
        if cache is None:
            validated[signature] = True
        return outputs

    def _execute(self, inputs, single=False, cache=None):
        if type(inputs) is not tuple:
            inputs = tuple(inputs[n] for n in self._input_names)
        old_cache = getattr(self, "_active_cache", None)
        self._active_cache = cache
        try:
            if _is_compiling():
                # Traced once per guard set, so the full checks cost nothing per
                # call there and dynamo turns them into guards.
                self._verify_state("build")
                outputs = self._run_checked(inputs, compiling=True, cache=cache)
                self._verify_state("forward")
                if single:
                    return next(iter(outputs.values()))
                return outputs
            generation = _watch_generation
            signature = self._signature(inputs)
            # _record_stores, compared: submodules and empty stores by identity,
            # the other stores by length and which values are None.
            same, copies, views, store_lens, tensor_ids, shape = self._stores
            try:
                if same != copies:
                    changed = True
                else:
                    changed = False
                    for v, expected_len in zip(views, store_lens):
                        if len(v) != expected_len:
                            changed = True
                            break
                    if not changed:
                        for t, expected_id in zip(_chain(views), tensor_ids):
                            if id(t) != expected_id:
                                dtypes_devices = [(x.dtype, x.device) if x is not None else None for x in _chain(views)]
                                current_shape = [*store_lens, *map(_is, _chain(views), _NONES), *dtypes_devices]
                                changed = current_shape != shape
                                break
            except Exception:  # noqa: BLE001 - a submodule's own __eq__ raised
                changed = True
            if changed or generation != self._state_seen or signature not in self._validated or cache is not None:
                outputs = self._validate(inputs, signature, generation, changed, cache=cache)
                if single:
                    return next(iter(outputs.values()))
                return outputs
            padding, program, output_slots = self._fast
            values = [*inputs, *padding]
            try:
                for module, ins, out, node, to_clear in program:
                    node_id = node[4]
                    args = (values[ins],) if type(ins) is int else tuple(ins(values)) if ins is not None else ()
                    if cache is not None and getattr(module, "_supports_inference_cache", False):
                        result = module(*args, cache=cache, node_id=node_id)
                    elif self.training and node_id in self.checkpoint_nodes:
                        if any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
                            result = torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)
                        else:
                            result = module(*args)
                    else:
                        result = (module(values[ins]) if type(ins) is int
                                  else module(*ins(values)) if ins is not None
                                  else module())
                    if out is not None and type(result) is _Tensor:
                        values[out] = result
                    else:
                        _, outs, out_ports, out_set, node_id = node
                        for slot, value in zip(outs, self._split_result(result, out_ports, out_set, node_id)):
                            values[slot] = value
                    for slot in to_clear:
                        values[slot] = None
            except Exception as error:
                ins, _, _, _, node_id = node
                leading = values[0]
                self._annotate_failure(node_id, error, [values[slot] for slot in ins],
                                       leading.shape[0] if leading.ndim else None, after_validation=True)
                raise
            if _watch_generation != generation:
                # Something registered state while this call ran --- perhaps one
                # of its own modules. Check it now, as the first call would have.
                generation = _watch_generation
                self._verify_state("forward")
                self._state_checked(generation)
            if single:
                return values[output_slots[0][1]]
            return OutputDict((name, values[slot]) for name, slot in output_slots)
        finally:
            self._active_cache = old_cache

    def _bind(self, args, kwargs):
        """Bind runtime tensors to the declared external inputs, in order."""
        names = self._input_names
        if not kwargs and len(args) == len(names):
            return args if type(args) is tuple else tuple(args)
        if not args and kwargs.keys() == self._input_set:
            return tuple(kwargs[n] for n in names)
        expected = "Expected exactly the external tensor inputs " + ", ".join(repr(n) for n in names)
        if len(args) > len(names):
            raise HNDLError("E_BINDING", expected)
        bound = dict(zip(names, args))
        for name, value in kwargs.items():
            if name in bound:
                raise HNDLError("E_BINDING", f"Input {name!r} was supplied twice")
            bound[name] = value
        if set(bound) != set(names):
            raise HNDLError("E_BINDING", expected)
        return tuple(bound[n] for n in names)

    def forward(
        self,
        *inputs: torch.Tensor,
        cache: InferenceCache | None = None,
        **named: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if "cache" in named and cache is None:
            cache = named.pop("cache")
        single = bool(inputs and not named and self._single_output)
        return self._execute(self._bind(inputs, named), single=single, cache=cache)

    def param_groups(self, weight_decay: float = 0.01) -> list[dict[str, Any]]:
        """Partition model parameters into weight-decay and no-decay groups."""
        all_params = list(self.parameters())
        if not all_params:
            return []

        no_decay_node_ops = {
            "rms_norm", "layer_norm", "group_norm", "batch_norm", "instance_norm",
            "adaptive_norm", "adaln_zero", "grn", "l2_norm", "dy_tanh",
            "embedding", "pos_embed", "cls_token", "learned_scale",
        }
        no_decay_param_names = {
            "bias", "A_log", "dt_bias", "D", "alpha", "learned_scale", "gamma", "beta",
        }

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        seen = set()
        for name, param in self.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))

            if param.ndim < 2:
                no_decay_params.append(param)
                continue

            base_name = name.split(".")[-1]
            if base_name in no_decay_param_names or base_name.endswith("bias"):
                no_decay_params.append(param)
                continue

            is_no_decay_node = False
            parts = name.split(".")
            for part in parts:
                if part.startswith("n_"):
                    nid = part[2:]
                    for n in self.plan.nodes:
                        if n.id == nid:
                            alias = self.plan._operation_name(n.op) if hasattr(self.plan, "_operation_name") else n.op.split("@")[0]
                            if alias in no_decay_node_ops:
                                is_no_decay_node = True
                            break

            if is_no_decay_node:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups

    def __getitem__(self, key: str | int | slice) -> nn.Module:
        if isinstance(key, str):
            try:
                return self.nodes[f"n_{key}"]
            except KeyError:
                for n in self.plan.nodes:
                    if n.source and n.source.get("binding") == key:
                        return self.nodes[f"n_{n.id}"]
                available = [n.id for n in self.plan.nodes]
                for n in self.plan.nodes:
                    if n.source and n.source.get("binding"):
                        available.append(n.source["binding"])
                import difflib

                matches = difflib.get_close_matches(key, available, n=1)
                suggestion = f". Did you mean {matches[0]!r}?" if matches else ""
                raise KeyError(f"{key}{suggestion}") from None
        if not self._chain:
            raise TypeError("Branched graphs support node-name lookup only")
        layers = tuple(self.nodes.values())
        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise ValueError(f"GraphModule slice step must be 1; got {key.step}")
            selected = tuple(self.plan.nodes)[key]
            return nn.Sequential(OrderedDict((f"n_{n.id}", self.nodes[f"n_{n.id}"]) for n in selected))
        if type(key) is int:
            return layers[key]
        raise TypeError("Layer index must be an integer, slice, or node name")

    def __len__(self) -> int:
        if not self._chain:
            raise TypeError("Branched graphs support node-name lookup only")
        return len(self.plan.nodes)

    def __iter__(self):
        if not self._chain:
            raise TypeError("Branched graphs support node-name lookup only")
        return iter(self.nodes.values())

    def _alias(self, op: str) -> str:
        registry = self.plan.registry
        try:
            return registry.by_identity(op).alias if registry is not None else op.split("@")[0]
        except HNDLError:
            return op.split("@")[0]

    def __repr__(self) -> str:
        lines = [contract_header(self.plan, type(self).__name__)]
        rows = [("index", "name", "operation", "input shape", "output shape")
                if self._chain else ("name", "operation", "input shapes", "output shapes")]
        for i, node in enumerate(self.plan.nodes):
            operation = self._alias(node.op)
            if self._chain:
                rows.append((str(i), node.id, operation,
                             _shape_text(next(iter(node.input_shapes.values()))),
                             _shape_text(next(iter(node.output_shapes.values())))))
            else:
                inputs = ", ".join(f"{p}={node.inputs[p]}:{_shape_text(s)}" for p, s in node.input_shapes.items())
                outputs = ", ".join(f"{p}={_shape_text(s)}" for p, s in node.output_shapes.items())
                rows.append((node.id, operation, inputs, outputs))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
                     for row in rows)
        return "\n".join(lines)


class Network(GraphModule):
    """Tensor facade over the graph: one tensor out, or a dict of named outputs."""

    def forward(
        self,
        *inputs: torch.Tensor,
        cache: InferenceCache | None = None,
        **named: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if "cache" in named and cache is None:
            cache = named.pop("cache")
        return self._execute(self._bind(inputs, named), single=self._single_output, cache=cache)


@contextmanager
def _initialization_rng(device, seed):
    if seed is None:
        yield
        return
    if type(seed) is not int:
        raise HNDLError("E_SCHEMA", "initialization_seed must be an integer or None")
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # torch.manual_seed seeds every CUDA device; only touch this scope.
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        yield


def _symbol_values(spec, node):
    """Resolved values of the shape symbols a module constructor requests."""
    values = {}
    shapes = {**node.input_shapes, **node.output_shapes}
    from .operator import ELLIPSIS, Sym
    for port in (*spec.inputs, *spec.outputs):
        if port.pattern is None or port.name not in shapes:
            continue
        shape = shapes[port.name]
        pattern = port.pattern
        if ELLIPSIS in pattern:
            index = pattern.index(ELLIPSIS)
            middle = len(shape) - (len(pattern) - 1)
            pattern = pattern[:index] + (None,) * middle + pattern[index + 1:]
        for axis, dimension in enumerate(pattern):
            if isinstance(dimension, Sym) and dimension.name not in values:
                values[dimension.name] = shape[axis] // dimension.scale
    return values


def construct(spec, node, device, dtype, initialization_seed=None):
    """Instantiate one operator's module from concrete plan arguments."""
    # A resolved node omits arguments added after release that hold their
    # default (``Arg(since=...)``); the module still receives every argument.
    kwargs = {name: arg.default for name, arg in spec.args.items() if arg.since is not None}
    kwargs.update(node.args)
    if spec.init_symbols:
        symbols = _symbol_values(spec, node)
        for name in spec.init_symbols:
            kwargs[name] = symbols[name]
    for name in spec.init_shapes:
        kwargs[name] = {port: tuple(shape) for port, shape in getattr(node, name).items()}
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    prev_dtype = torch.get_default_dtype()
    set_default = (
        initialization_seed is None
        and dtype is not None
        and hasattr(dtype, "is_floating_point")
        and dtype.is_floating_point
        and dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    )
    if set_default and dtype != prev_dtype:
        torch.set_default_dtype(dtype)
    try:
        if device_obj.type == "cpu" and getattr(torch._C, "_len_torch_function_stack", lambda: 1)() == 0:
            layer = spec.module(**kwargs)
        else:
            with torch.device(device_obj):
                layer = spec.module(**kwargs)
    finally:
        if set_default and dtype != prev_dtype:
            torch.set_default_dtype(prev_dtype)
    if not isinstance(layer, nn.Module):
        raise HNDLError("E_REGISTRY", f"{node.id}: operator {spec.alias} did not construct an nn.Module")
    if device_obj.type == "meta":
        # The allocation-free probe only measures; never move a module onto meta.
        return layer
    return layer.to(device=device_obj, dtype=dtype)


def _state_bytes(module, compute_dtype=None):
    """Unique registered parameter/buffer storage, including nonpersistent buffers.

    With ``compute_dtype`` the count assumes floating tensors will be cast to
    it, which is how the allocation-free probe predicts the real footprint.
    """
    storages = {}
    for value in (*module.parameters(), *module.buffers()):
        storage = value.untyped_storage()
        key = (value.device, storage._cdata) if value.device.type == "meta" else (value.device, storage.data_ptr())
        if compute_dtype is not None and value.is_floating_point():
            elem_size = torch.finfo(compute_dtype).bits // 8
            nbytes = (storage.nbytes() // value.element_size()) * elem_size
        else:
            nbytes = storage.nbytes()
        storages[key] = max(storages.get(key, 0), nbytes)
    return sum(storages.values())


def parameter_counts(
    plan: ResolvedPlan,
    *,
    registry: Registry | None = None,
) -> dict[str, int | None]:
    """Per-node parameter counts from an allocation-free meta-device construction."""
    from .registry import Registry
    registry = registry if registry is not None else plan.registry or Registry.builtins()
    counts: dict[str, int | None] = {}
    with registry.activated():
        for node in plan.nodes:
            spec = registry.by_identity(node.op)
            try:
                layer = construct(spec, node, "meta", DTYPES[plan.dtype])
            except Exception:
                counts[node.id] = None
                continue
            counts[node.id] = sum(p.numel() for p in layer.parameters())
    return counts


INITIALIZERS = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "truncated_normal": nn.init.trunc_normal_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
    "orthogonal": nn.init.orthogonal_,
}


def _initializer_identity(value):
    """Compare two overrides exactly, distinguishing +0.0 from -0.0."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, item.hex() if type(item) is float else item)
                            for key, item in value.items()))
    # float.hex distinguishes +0.0 and -0.0, which fill_ preserves.
    return value.hex()


def _apply_initializer(parameter, override, name, node_id):
    """Fill one parameter in place; scheme draws come from the active RNG."""
    if not isinstance(override, Mapping):
        parameter.fill_(override)
        return
    kind = override["kind"]
    arguments = {key: item for key, item in override.items() if key != "kind"}
    try:
        INITIALIZERS[kind](parameter, **arguments)
    except (ValueError, RuntimeError) as exc:
        raise HNDLError("E_INITIALIZATION",
                        f"Initializer {kind} cannot initialize parameter {name!r} with shape "
                        f"{tuple(parameter.shape)}: {exc}", node=node_id) from None


def _prepare_parameter_settings(layer, node, dtype):
    """Validate exact targets and aliases without changing any parameter."""
    parameters = dict(layer.named_parameters(remove_duplicate=False))
    buffers = dict(layer.named_buffers(remove_duplicate=False))
    initialization = node.initialization["overrides"]
    trainability = node.trainability["overrides"]
    default_trainability = node.trainability["default"]

    for code, assignments in (("E_INITIALIZATION", initialization), ("E_TRAINABILITY", trainability)):
        for name in sorted(assignments):
            if name not in parameters:
                if name in buffers:
                    detail = "is a buffer, not a parameter"
                else:
                    import difflib

                    matches = difflib.get_close_matches(name, parameters.keys(), n=1)
                    suggestion = f". Did you mean {matches[0]!r}?" if matches else ""
                    detail = f"is not a registered parameter{suggestion}"
                raise HNDLError(code, f"Parameter target {name!r} {detail}", node=node.id)

    by_id, names, storage_groups, buffer_storages = {}, {}, {}, set()
    for name, parameter in parameters.items():
        identity = id(parameter)
        by_id[identity] = parameter
        names.setdefault(identity, []).append(name)
        storage = parameter.untyped_storage()
        if storage.nbytes():
            key = (parameter.device, storage.data_ptr())
            storage_groups.setdefault(key, set()).add(identity)
    for buffer in buffers.values():
        storage = buffer.untyped_storage()
        if storage.nbytes():
            buffer_storages.add((buffer.device, storage.data_ptr()))

    constants, flags = {}, {}
    for assignments, destination, code in (
        (initialization, constants, "E_INITIALIZATION"),
        (trainability, flags, "E_TRAINABILITY"),
    ):
        for name in sorted(assignments):
            identity = id(parameters[name])
            value = assignments[name]
            previous = destination.get(identity)
            if identity in destination and (
                _initializer_identity(previous) != _initializer_identity(value)
                if code == "E_INITIALIZATION" else previous != value
            ):
                raise HNDLError(code, f"Conflicting settings for aliases of parameter {names[identity]!r}", node=node.id)
            destination[identity] = value

    effective_flags = {identity: flags.get(identity, parameter.requires_grad if default_trainability is None
                                           else default_trainability)
                       for identity, parameter in by_id.items()}
    for key, identities in storage_groups.items():
        if identities & constants.keys():
            if len(identities) > 1:
                raise HNDLError("E_INITIALIZATION", "Constant override targets distinct parameters sharing storage; use one Parameter with named aliases instead", node=node.id)
            if key in buffer_storages:
                raise HNDLError("E_INITIALIZATION", "Constant override targets parameter storage also registered as a buffer", node=node.id)
        touched = default_trainability is not None or bool(identities & flags.keys())
        if touched and len({effective_flags[identity] for identity in identities}) > 1:
            raise HNDLError("E_TRAINABILITY", "Parameters sharing storage have conflicting effective trainability settings", node=node.id)

    result = []
    for identity, parameter in by_id.items():
        constant = constants.get(identity)
        flag = effective_flags[identity] if default_trainability is not None or identity in flags else None
        target = names[identity][0]
        if constant is not None and parameter.dtype != dtype:
            kind = "constant" if not isinstance(constant, Mapping) else f"{constant['kind']}"
            raise HNDLError("E_INITIALIZATION", f"Parameter {target!r} has unsupported {kind}-initialization dtype {parameter.dtype}; expected {dtype}", node=node.id)
        # Rank is checked before any node is touched so a rejected scheme never
        # leaves an earlier node's parameters half initialized.
        if isinstance(constant, Mapping) and constant["kind"] in MATRIX_SCHEMES and parameter.dim() < 2:
            raise HNDLError("E_INITIALIZATION", f"Initializer {constant['kind']} requires a parameter of at least two dimensions; {target!r} has shape {tuple(parameter.shape)}", node=node.id)
        if flag is True and not (parameter.is_floating_point() or parameter.is_complex()):
            raise HNDLError("E_TRAINABILITY", f"Parameter {target!r} with dtype {parameter.dtype} cannot require gradients", node=node.id)
        result.append((parameter, target, constant, flag))
    return result


def _get_parameter_or_buffer(module: nn.Module, name: str) -> torch.Tensor:
    curr = module
    for part in name.split("."):
        curr = getattr(curr, part)
    return curr


def _set_parameter_or_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    parts = name.split(".")
    curr = module
    for part in parts[:-1]:
        curr = getattr(curr, part)
    setattr(curr, parts[-1], tensor)


def _apply_depth_scaled_init(plan: ResolvedPlan, modules: Mapping[str, nn.Module]) -> None:
    """Scale residual branch exit projection weights by 1 / sqrt(2 * N_layers)."""
    add_nodes = [n for n in plan.nodes if n.op.split("@")[0] == "add"]
    if not add_nodes:
        return

    node_by_id = {n.id: n for n in plan.nodes}
    producer_of: dict[str, str] = {}
    for n in plan.nodes:
        for port in n.output_shapes:
            producer_of[f"node:{n.id}/{port}"] = n.id
            producer_of[f"{n.id}:{port}"] = n.id
        producer_of[n.id] = n.id

    def get_ancestors(node_id: str) -> set[str]:
        ancestors = set()
        queue = [node_id]
        while queue:
            curr = queue.pop(0)
            if curr in node_by_id:
                for ref in node_by_id[curr].inputs.values():
                    parent = producer_of.get(ref)
                    if parent and parent not in ancestors:
                        ancestors.add(parent)
                        queue.append(parent)
        return ancestors

    def has_params(nid: str) -> bool:
        mod = modules.get(f"n_{nid}")
        return mod is not None and any(p.numel() > 0 for p in mod.parameters())

    def find_exit_projection(start_node_id: str, shortcut_ancestors: set[str]) -> str | None:
        curr = start_node_id
        while curr:
            if has_params(curr):
                return curr
            c_node = node_by_id.get(curr)
            if not c_node or not c_node.inputs:
                break
            next_curr = None
            for ref in c_node.inputs.values():
                parent = producer_of.get(ref)
                if parent and parent not in shortcut_ancestors:
                    next_curr = parent
                    break
            if next_curr is None:
                ref = next(iter(c_node.inputs.values()))
                next_curr = producer_of.get(ref)
            curr = next_curr
        return None

    exit_projections: list[str] = []
    for add_node in add_nodes:
        refs = list(add_node.inputs.values())
        if len(refs) != 2:
            continue
        p0 = producer_of.get(refs[0])
        p1 = producer_of.get(refs[1])
        if p0 is None and p1 is not None:
            exit_node = find_exit_projection(p1, set())
            if exit_node:
                exit_projections.append(exit_node)
        elif p1 is None and p0 is not None:
            exit_node = find_exit_projection(p0, set())
            if exit_node:
                exit_projections.append(exit_node)
        elif p0 is not None and p1 is not None:
            anc0 = get_ancestors(p0)
            anc1 = get_ancestors(p1)
            if p0 in anc1:
                exit_node = find_exit_projection(p1, anc0 | {p0})
                if exit_node:
                    exit_projections.append(exit_node)
            elif p1 in anc0:
                exit_node = find_exit_projection(p0, anc1 | {p1})
                if exit_node:
                    exit_projections.append(exit_node)
            else:
                if has_params(p1) and not has_params(p0):
                    exit_node = find_exit_projection(p1, anc0 | {p0})
                elif has_params(p0) and not has_params(p1):
                    exit_node = find_exit_projection(p0, anc1 | {p1})
                else:
                    exit_node = find_exit_projection(p1, set()) or find_exit_projection(p0, set())
                if exit_node:
                    exit_projections.append(exit_node)

    num_layers = len(exit_projections)
    if num_layers == 0:
        return

    import math
    lambda_factor = 1.0 / math.sqrt(2.0 * num_layers)

    for nid in exit_projections:
        layer = modules.get(f"n_{nid}")
        if layer is None:
            continue
        with torch.no_grad():
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.mul_(lambda_factor)
            elif hasattr(layer, "o_proj") and getattr(layer.o_proj, "weight", None) is not None:
                layer.o_proj.weight.mul_(lambda_factor)
            elif hasattr(layer, "out_proj") and getattr(layer.out_proj, "weight", None) is not None:
                layer.out_proj.weight.mul_(lambda_factor)
            elif hasattr(layer, "conv3") and getattr(layer.conv3, "weight", None) is not None:
                layer.conv3.weight.mul_(lambda_factor)
            elif hasattr(layer, "conv2") and getattr(layer.conv2, "weight", None) is not None:
                layer.conv2.weight.mul_(lambda_factor)
            else:
                for p in reversed(list(layer.parameters())):
                    if p.ndim >= 2:
                        p.mul_(lambda_factor)
                        break


def _build(plan, *, device, initialization_seed=None, registry=None, facade=False, limits=None, checkpoint=False, init_policy="default"):
    if plan.dtype not in ("float32", "float16", "bfloat16"):
        raise HNDLError("E_SCHEMA", f"Unsupported plan dtype {plan.dtype!r}")
    dtype = DTYPES[plan.dtype]
    device = torch.device(device)
    if device.type == "cpu":
        device = torch.device("cpu")
    if device.type not in ("cpu", "cuda"):
        raise HNDLError("E_SCHEMA", f"Unsupported execution device {device}; use cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise HNDLError("E_RUNTIME", "CUDA was requested but is unavailable")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if device.index >= torch.cuda.device_count():
            raise HNDLError("E_RUNTIME", f"CUDA device {device.index} is unavailable")
    registry = registry if registry is not None else plan.registry
    if registry is None:
        from .registry import Registry
        registry = Registry.builtins()
    from .resolver import validate_concrete_plan, _limits
    bounds = _limits(limits)
    limits_items = tuple(sorted(limits.items())) if isinstance(limits, dict) else ()
    bounds_items = tuple(sorted(bounds.items()))
    stamp = getattr(plan, "_validated_stamp", None)
    if stamp != (id(registry), limits_items) and stamp != (id(registry), bounds_items):
        plan = validate_concrete_plan(plan, registry=registry, limits=limits)
    specs = {node.id: registry.by_identity(node.op) for node in plan.nodes}
    port_orders = {node.id: specs[node.id].input_ports_for(node.args) for node in plan.nodes}

    if checkpoint is True:
        HEAVY_BLOCKS = {
            "transformer_block", "moe", "attention", "cross_attention",
            "latent_attention", "gated_delta_attention", "joint_attention", "mamba2"
        }
        checkpoint_nodes = {
            node.id for node in plan.nodes
            if (node.op.split("@")[0] in HEAVY_BLOCKS
                or getattr(specs[node.id], "alias", "") in HEAVY_BLOCKS
                or getattr(specs[node.id], "identity", "") in HEAVY_BLOCKS)
        }
    elif isinstance(checkpoint, (Sequence, set)) and not isinstance(checkpoint, str):
        checkpoint_nodes = set(checkpoint)
    elif checkpoint is False or checkpoint is None:
        checkpoint_nodes = set()
    else:
        raise HNDLError("E_ARGUMENT", f"Invalid checkpoint argument {checkpoint!r}; expected bool or sequence of node IDs")

    if init_policy not in ("default", "depth_scaled"):
        raise HNDLError("E_ARGUMENT", f"Unknown init_policy {init_policy!r}; expected 'default' or 'depth_scaled'")

    # Bound registered storage with an allocation-free meta construction first.
    total = 0
    unbounded = []
    with registry.activated():
        for node in plan.nodes:
            try:
                probe = construct(specs[node.id], node, "meta", dtype)
            except Exception:
                unbounded.append(node.id)
                continue
            total += _state_bytes(probe, dtype)
            if total > bounds["max_state_bytes"]:
                raise HNDLError("E_RESOURCE", "Plan exceeds max_state_bytes before allocation", node=node.id)

    modules = OrderedDict()
    module_owners, tensor_owners, storage_owners = {}, {}, {}

    def check_ownership(layer, node_id, tied_tensors=()):
        owned = [(module_owners, id(child)) for child in layer.modules()]
        for tensor in (*layer.parameters(), *layer.buffers()):
            if tensor in tied_tensors:
                continue
            owned.append((tensor_owners, id(tensor)))
            storage = tensor.untyped_storage()
            if storage.nbytes():
                owned.append((storage_owners, (tensor.device, storage.data_ptr())))
        for owners, key in owned:
            if key in owners and owners[key] != node_id:
                raise HNDLError("E_REGISTRY", f"{node_id}: operator reuses a module or registered state from node {owners[key]}; each node must own an independent instance")
        for owners, key in owned:
            owners[key] = node_id

    with registry.activated(), _initialization_rng(device, initialization_seed):
        for node in plan.nodes:
            layer = construct(specs[node.id], node, device, dtype, initialization_seed=initialization_seed)
            tie = node.source.get("tie") if node.source else None
            tied_tensors = set()
            if tie:
                for target_p, source_ref in tie.items():
                    if "." not in source_ref:
                        raise HNDLError("E_SCHEMA", f"Tied parameter source reference must be 'node_id.param', got {source_ref!r}", node=node.id)
                    source_node_id, source_p = source_ref.split(".", 1)
                    source_key = f"n_{source_node_id}"
                    if source_key not in modules:
                        raise HNDLError("E_SCHEMA", f"Tied parameter source node {source_node_id!r} not found or not preceding {node.id}", node=node.id)
                    source_layer = modules[source_key]
                    try:
                        source_tensor = _get_parameter_or_buffer(source_layer, source_p)
                    except AttributeError:
                        raise HNDLError("E_SCHEMA", f"Tied parameter source {source_ref} does not exist", node=node.id)
                    try:
                        target_tensor = _get_parameter_or_buffer(layer, target_p)
                    except AttributeError:
                        raise HNDLError("E_SCHEMA", f"Tied parameter target {target_p} does not exist on {node.id}", node=node.id)
                    if source_tensor.shape != target_tensor.shape:
                        raise HNDLError("E_SCHEMA", f"Tied parameter shape mismatch: {source_ref} shape {list(source_tensor.shape)} != {target_p} shape {list(target_tensor.shape)}", node=node.id)
                    if source_tensor.dtype != target_tensor.dtype:
                        raise HNDLError("E_SCHEMA", f"Tied parameter dtype mismatch: {source_ref} dtype {source_tensor.dtype} != {target_p} dtype {target_tensor.dtype}", node=node.id)
                    _set_parameter_or_buffer(layer, target_p, source_tensor)
                    tied_tensors.add(source_tensor)
            check_ownership(layer, node.id, tied_tensors=tied_tensors)
            if node.id in unbounded:
                total += _state_bytes(layer)
                if total > bounds["max_state_bytes"]:
                    raise HNDLError("E_RESOURCE", "Plan exceeds max_state_bytes", node=node.id)
            modules[f"n_{node.id}"] = layer
        # Resolve every target before applying settings to any graph node.
        settings = [_prepare_parameter_settings(modules[f"n_{node.id}"], node, dtype) for node in plan.nodes]
        with torch.no_grad():
            for node, assignments in zip(plan.nodes, settings):
                for parameter, target, override, trainable in assignments:
                    if override is not None:
                        _apply_initializer(parameter, override, target, node.id)
                    if trainable is not None:
                        parameter.requires_grad_(trainable)
                modules[f"n_{node.id}"].train(modules[f"n_{node.id}"].training)

        if init_policy == "depth_scaled":
            _apply_depth_scaled_init(plan, modules)

    receipt = {"torch_version": str(torch.__version__), "device": str(device), "dtype": plan.dtype,
               "initialization_seed": initialization_seed,
               "seed_mode": "caller" if initialization_seed is None else "isolated",
               "state_bytes": total}
    cls = Network if facade else GraphModule
    return cls(plan, modules, device, receipt, port_orders, checkpoint_nodes=checkpoint_nodes)


def build(
    plan: ResolvedPlan,
    *,
    device: str | torch.device,
    initialization_seed: int | None = None,
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    checkpoint: bool | Sequence[str] = False,
    init_policy: str = "default",
) -> GraphModule:
    """Build a concrete plan; forward accepts named inputs and returns a dict."""
    return _build(
        plan,
        device=device,
        initialization_seed=initialization_seed,
        registry=registry,
        limits=limits,
        checkpoint=checkpoint,
        init_policy=init_policy,
    )


def _network(
    resolve_name: str,
    source: Any,
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    device: str | torch.device,
    dtype: str = "float32",
    registry: Registry | None = None,
    initialization_seed: int | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
    checkpoint: bool | Sequence[str] = False,
    init_policy: str = "default",
) -> Network:
    import hndl
    resolve = getattr(hndl, resolve_name)
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape,
                   dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return _build(
        plan,
        device=device,
        initialization_seed=initialization_seed,
        registry=registry,
        facade=True,
        limits=limits,
        checkpoint=checkpoint,
        init_policy=init_policy,
    )


def network(
    source: str,
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    device: str | torch.device,
    dtype: str = "float32",
    registry: Registry | None = None,
    initialization_seed: int | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
    checkpoint: bool | Sequence[str] = False,
    init_policy: str = "default",
) -> Network:
    """Resolve declarative source and return a tensor-in/tensor-out module."""
    return _network(
        "resolve",
        source,
        input_shape=input_shape,
        output_shape=output_shape,
        device=device,
        dtype=dtype,
        registry=registry,
        initialization_seed=initialization_seed,
        limits=limits,
        input_dtype=input_dtype,
        checkpoint=checkpoint,
        init_policy=init_policy,
    )


def network_file(
    path: str | os.PathLike[str],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    device: str | torch.device,
    dtype: str = "float32",
    registry: Registry | None = None,
    initialization_seed: int | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
    checkpoint: bool | Sequence[str] = False,
    init_policy: str = "default",
) -> Network:
    """Read declarative UTF-8 source and construct its resolved network."""
    return _network(
        "resolve_file",
        path,
        input_shape=input_shape,
        output_shape=output_shape,
        device=device,
        dtype=dtype,
        registry=registry,
        initialization_seed=initialization_seed,
        limits=limits,
        input_dtype=input_dtype,
        checkpoint=checkpoint,
        init_policy=init_policy,
    )


def network_from_callable(
    fn: Callable[..., Any],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    device: str | torch.device,
    dtype: str = "float32",
    registry: Registry | None = None,
    initialization_seed: int | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
    checkpoint: bool | Sequence[str] = False,
    init_policy: str = "default",
) -> Network:
    """Capture trusted Python exactly once, then construct the resolved network."""
    return _network(
        "resolve_callable",
        fn,
        input_shape=input_shape,
        output_shape=output_shape,
        device=device,
        dtype=dtype,
        registry=registry,
        initialization_seed=initialization_seed,
        limits=limits,
        input_dtype=input_dtype,
        checkpoint=checkpoint,
        init_policy=init_policy,
    )


def load_weights(
    model: GraphModule | Network,
    source: str | Path | dict[str, torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    key_map: Mapping[str, str] | Callable[[str], str] | None = None,
    strict: bool = True,
) -> None:
    """Load external checkpoint state dict into an HNDL model with key remapping."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file  # type: ignore
                state_dict = load_file(str(path))
            except ImportError:
                raise HNDLError("E_RUNTIME", "safetensors package required to load .safetensors files")
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
    elif isinstance(source, Mapping):
        state_dict = dict(source)
    else:
        raise TypeError(f"source must be a path or state dict Mapping, got {type(source).__name__}")

    target_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in state_dict.items():
        if key_map is not None:
            if callable(key_map):
                new_k = key_map(k)
            else:
                new_k = k
                for prefix, repl in key_map.items():
                    if k.startswith(prefix):
                        new_k = repl + k[len(prefix):]
                        break
        else:
            new_k = k

        if new_k in target_keys:
            remapped[new_k] = v
        elif f"nodes.n_{new_k}" in target_keys:
            remapped[f"nodes.n_{new_k}"] = v
        elif f"nodes.{new_k}" in target_keys:
            remapped[f"nodes.{new_k}"] = v
        else:
            remapped[new_k] = v

    with model.allow_state_mutation():
        model.load_state_dict(remapped, strict=strict)

