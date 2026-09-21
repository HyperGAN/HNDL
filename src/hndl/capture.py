"""Torch-free symbolic graph capture for trusted authoring functions."""

from __future__ import annotations

from contextvars import ContextVar
import re

from .errors import HNDLError
from .registry import Registry, normalize_arguments
from .settings import normalize_settings
from .types import Graph, Node


_ACTIVE: ContextVar[Capture | None] = ContextVar("hndl_capture", default=None)
_DEFAULT = object()
_NODE_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


class Symbol:
    """An edge in one capture; never an eager tensor or a tensor value."""

    __slots__ = ("_owner", "ref")

    def __init__(self, owner: Capture, ref: str):
        self._owner = owner
        self.ref = ref

    def __repr__(self):
        return f"Symbol({self.ref!r})"

    def _unsupported(self, *args, **kwargs):
        raise HNDLError("E_CAPTURE", "Symbolic tensors cannot be evaluated as Python or eager tensor values")

    __bool__ = __int__ = __float__ = __index__ = __iter__ = __len__ = _unsupported
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _unsupported
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _unsupported
    __truediv__ = __rtruediv__ = __matmul__ = __rmatmul__ = _unsupported
    __getitem__ = __neg__ = __pos__ = _unsupported

    @classmethod
    def __torch_function__(cls, *args, **kwargs):
        raise HNDLError("E_CAPTURE", "Use registered symbolic operators instead of eager torch operations")


def _error(code, message, source=None):
    location = {} if source is None else {key: source[key] for key in ("line", "column") if key in source}
    return HNDLError(code, message, **location)


class Capture:
    """One scoped authoring transaction, shared by the two frontends."""

    def __init__(self, *, input_shape, output_shape, dtype="float32", registry=None,
                 frontend="python_callable@1", limits=None, input_dtype=None):
        self.registry = Registry.builtins() if registry is None else registry
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.dtype = dtype
        self.input_dtype = input_dtype
        self.frontend = frontend
        self.nodes: list[Node] = []
        self._ids: set[str] = set()
        self.input = Symbol(self, "input:x")
        self.current: Symbol | None = self.input
        self.max_nodes = 4096 if limits is None else limits.get("max_nodes", 4096)
        if type(self.max_nodes) is not int or self.max_nodes < 1:
            raise HNDLError("E_RESOURCE", "max_nodes must be a positive integer")
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.reset(self._token)
        self._token = None

    def _symbol(self, value, source=None):
        if not isinstance(value, Symbol) or value._owner is not self:
            raise _error("E_CAPTURE", "Input must be a symbolic tensor from this capture", source)
        return value

    def emit(self, op, positional, keyword, *, source=None):
        # Factories carry exact identities, never whichever alias is newest.
        registered = self.registry.by_identity(op.key)
        if registered.key != op.key:
            raise _error("E_CAPTURE", "Operator is not present in this capture registry", source)
        args = list(positional)
        kwargs = dict(keyword)
        name = kwargs.pop("name", _DEFAULT)
        policy = kwargs.pop("policy", _DEFAULT)
        construction = {key: kwargs.pop(key) for key in ("init", "trainable") if key in kwargs}
        try:
            initialization, trainability = normalize_settings(**construction)
        except HNDLError as exc:
            if source is not None and exc.line is None:
                raise _error(exc.code, exc.message, source) from None
            raise
        if policy is not _DEFAULT and type(policy) is not str:
            raise _error("E_ARGUMENT", "policy must be a registered policy name", source)
        policy = None if policy is _DEFAULT else policy
        ports = tuple(op.input_ports)
        bindings = {}

        if op.variadic is not None:
            prefix = op.variadic
            count = 0
            while args and isinstance(args[0], Symbol):
                bindings[f"{prefix}{count}"] = self._symbol(args.pop(0), source)
                count += 1
            for key in tuple(kwargs):
                if re.fullmatch(re.escape(prefix) + r"[0-9]+", key):
                    if key in bindings:
                        raise _error("E_BINDING", f"Duplicate input binding {key}", source)
                    bindings[key] = self._symbol(kwargs.pop(key), source)
            if len(bindings) < 2 or set(bindings) != {f"{prefix}{i}" for i in range(len(bindings))}:
                raise _error("E_BINDING", f"Variadic inputs require contiguous {prefix}0...{prefix}N bindings with at least two tensors", source)
            if "input_count" in kwargs and (type(kwargs["input_count"]) is not int
                                             or kwargs["input_count"] != len(bindings)):
                raise _error("E_BINDING", "input_count does not match supplied tensors", source)
            kwargs["input_count"] = len(bindings)
        elif len(ports) == 1:
            port = ports[0]
            if args and isinstance(args[0], Symbol):
                bindings[port] = self._symbol(args.pop(0), source)
            if port in kwargs:
                if port in bindings:
                    raise _error("E_BINDING", f"Duplicate input binding {port}", source)
                bindings[port] = self._symbol(kwargs.pop(port), source)
            if port not in bindings:
                if self.current is None:
                    raise _error("E_CURRENT", "No current tensor; supply an explicit input after a multi-output operation", source)
                bindings[port] = self.current
        else:
            for port in ports:
                if args and isinstance(args[0], Symbol):
                    bindings[port] = self._symbol(args.pop(0), source)
                if port in kwargs:
                    if port in bindings:
                        raise _error("E_BINDING", f"Duplicate input binding {port}", source)
                    bindings[port] = self._symbol(kwargs.pop(port), source)
                if port not in bindings:
                    raise _error("E_BINDING", f"Multi-input operation requires explicit port {port}", source)

        if name is _DEFAULT:
            name = f"n{len(self.nodes)}"
        if type(name) is not str or _NODE_ID.fullmatch(name) is None:
            raise _error("E_NAME", "Node names must match [a-z][a-z0-9_]*", source)
        if name in self._ids:
            raise _error("E_NAME", f"Duplicate node name {name!r}", source)
        if len(self.nodes) >= self.max_nodes:
            raise _error("E_RESOURCE", "Graph exceeds max_nodes", source)

        explicit = set(kwargs)
        if args:
            explicit.update((op.positional_rest,) if op.positional_rest is not None else op.positional_names[:len(args)])
        try:
            normalized = normalize_arguments(op, tuple(args), kwargs, policy=policy)
        except HNDLError as exc:
            if source is not None and exc.line is None:
                raise _error(exc.code, exc.message, source) from None
            raise
        profile = op.policies[policy] if policy else None
        policy_fields = set(profile.requires) if profile else set()
        metadata = dict(source or {})
        metadata["argument_origins"] = {
            key: "explicit" if key in explicit else "policy-selected" if key in policy_fields else "operator default"
            for key in normalized
        }
        if profile:
            metadata["policy"] = profile.identity
        node = Node(id=name, op=op.key, args=normalized,
                    inputs={port: symbol.ref for port, symbol in bindings.items()},
                    outputs=tuple(op.output_ports), source=metadata,
                    initialization=initialization, trainability=trainability)
        self.nodes.append(node)
        self._ids.add(name)
        outputs = tuple(Symbol(self, f"node:{name}/{port}") for port in op.output_ports)
        self.current = outputs[0] if len(outputs) == 1 else None
        return outputs[0] if len(outputs) == 1 else outputs

    def finish(self, selected=_DEFAULT):
        if selected is _DEFAULT:
            if self.current is None:
                raise HNDLError("E_CURRENT", "The final operation has no current tensor; select one output explicitly")
            selected = self.current
        if not isinstance(selected, Symbol):
            raise HNDLError("E_OUTPUT", "The selected output must be one symbolic tensor")
        self._symbol(selected)
        return Graph(nodes=tuple(self.nodes), input_shape=self.input_shape,
                     output_shape=self.output_shape, output_ref=selected.ref,
                     dtype=self.dtype, frontend=self.frontend, input_dtype=self.input_dtype)


class OperatorNamespace:
    """Attribute factories with fixed operator identities, not live modules."""

    def __init__(self, registry=None):
        self._registry = registry

    def __getattr__(self, alias):
        if alias.startswith("_"):
            raise AttributeError(alias)
        registry = Registry.builtins() if self._registry is None else self._registry
        op = registry.get(alias)

        def operation(*args, **kwargs):
            capture = _ACTIVE.get()
            if capture is None:
                raise HNDLError("E_CAPTURE", "Symbolic operations require an active authoring capture")
            return capture.emit(op, args, kwargs)

        operation.__name__ = alias
        return operation


ops = OperatorNamespace()


def capture_callable(fn, *, input_shape, output_shape, dtype="float32", registry=None, limits=None,
                     input_dtype=None):
    """Invoke trusted Python exactly once and return its finite author graph."""
    if not callable(fn):
        raise TypeError("resolve_callable requires a callable, not source text")
    with Capture(input_shape=input_shape, output_shape=output_shape, dtype=dtype,
                 registry=registry, limits=limits, input_dtype=input_dtype) as capture:
        result = fn(capture.input)
        return capture.finish(_DEFAULT if result is None else result)


def resolve_callable(fn, *, input_shape, output_shape, dtype="float32", registry=None, limits=None,
                     input_dtype=None):
    from .resolver import resolve_graph

    registry = Registry.builtins() if registry is None else registry
    graph = capture_callable(fn, input_shape=input_shape, output_shape=output_shape,
                             dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return resolve_graph(graph, registry=registry, limits=limits)
