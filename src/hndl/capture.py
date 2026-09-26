"""Torch-free symbolic graph capture for trusted authoring functions."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
import re

from .errors import HNDLError, source_location
from .registry import Registry, normalize_arguments
from .resolver import _limits
from .settings import normalize_settings
from .types import (EXTERNAL_INPUT, EXTERNAL_OUTPUT, Graph, Node, named_contracts,
                    named_dtypes)


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
    return HNDLError(code, message, **source_location(source))


def _foreign_operator(registry, op):
    if op.key in registry._identities:
        problem = "declares that identity with a different implementation"
    else:
        problem = "does not hold it"
    hint = "pass the registry that declares it to resolve_callable or network_from_callable"
    if op.alias in registry.aliases:
        hint += f", or call ops.{op.alias} to use the capture's own binding"
    return (f"{op.alias} was taken from another registry's ops and binds {op.key}, but this "
            f"capture's registry {problem}; {hint}")


class Capture:
    """One scoped authoring transaction, shared by the two frontends."""

    def __init__(self, *, input_shape, output_shape, dtype="float32", registry=None,
                 frontend="python_callable@1", limits=None, input_dtype=None):
        self.registry = Registry.builtins() if registry is None else registry
        shapes, self.named_inputs = named_contracts(input_shape, "input_shape", EXTERNAL_INPUT)
        dtypes = named_dtypes(input_dtype, tuple(shapes))
        self.input_contracts = {name: {"shape": shape, "dtype": dtypes[name] or dtype}
                                for name, shape in shapes.items()}
        self.output_contracts, self.named_outputs = named_contracts(output_shape, "output_shape", EXTERNAL_OUTPUT)
        reserved = sorted((set(self.input_contracts) | set(self.output_contracts)) & set(self.registry.aliases))
        if reserved:
            raise HNDLError("E_NAME", f"External port names collide with operator aliases: {', '.join(reserved)}")
        self.input_shape = next(iter(self.input_contracts.values()))["shape"]
        self.output_shape = next(iter(self.output_contracts.values()))
        self.dtype = dtype
        self.input_dtype = input_dtype
        self.frontend = frontend
        self.nodes: list[Node] = []
        self._ids: set[str] = set()
        self.inputs = {name: Symbol(self, f"input:{name}") for name in self.input_contracts}
        self.input = next(iter(self.inputs.values()))
        self.current: Symbol | None = self.input
        self.max_nodes = _limits(limits)["max_nodes"]
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
        # ``registry.ops`` binds a declaration when the attribute is read; it
        # may run in any capture whose registry holds that very declaration,
        # as every registry built from ``Registry.builtins()`` does for the
        # built-ins, and nowhere else.
        if self.registry._identities.get(op.key) is not op:
            raise _error("E_CAPTURE", _foreign_operator(self.registry, op), source)
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
        # A variadic output operator fixes its port count from its own
        # arguments, so the ports come from the normalized call, not the
        # declaration alone.
        output_ports = op.output_ports_for(normalized)
        node = Node(id=name, op=op.key, args=normalized,
                    inputs={port: symbol.ref for port, symbol in bindings.items()},
                    outputs=output_ports, source=metadata,
                    initialization=initialization, trainability=trainability)
        self.nodes.append(node)
        self._ids.add(name)
        outputs = tuple(Symbol(self, f"node:{name}/{port}") for port in output_ports)
        if op.returns_tuple(normalized):
            self.current = None
            return outputs
        self.current = outputs[0]
        return outputs[0]

    def finish(self, selected=_DEFAULT):
        if self.named_outputs:
            if not isinstance(selected, Mapping):
                raise HNDLError("E_OUTPUT", "Named outputs require one symbolic tensor per declared name: "
                                            + ", ".join(self.output_contracts))
            declared, supplied = set(self.output_contracts), set(selected)
            if declared != supplied:
                details = []
                if declared - supplied:
                    details.append("never selected: " + ", ".join(sorted(declared - supplied)))
                if supplied - declared:
                    details.append("undeclared: " + ", ".join(sorted(supplied - declared)))
                raise HNDLError("E_OUTPUT", f"Declared outputs are {', '.join(self.output_contracts)}; "
                                            + "; ".join(details))
            chosen = {}
            for name in self.output_contracts:
                value = selected[name]
                if not isinstance(value, Symbol):
                    raise HNDLError("E_OUTPUT", f"Output {name!r} must be one symbolic tensor")
                chosen[name] = self._symbol(value)
        else:
            if selected is _DEFAULT:
                if self.current is None:
                    raise HNDLError("E_CURRENT", "The final operation has no current tensor; select one output explicitly")
                selected = self.current
            if not isinstance(selected, Symbol):
                raise HNDLError("E_OUTPUT", "The selected output must be one symbolic tensor")
            chosen = {EXTERNAL_OUTPUT: self._symbol(selected)}
        outputs = {name: {"ref": symbol.ref, "shape": self.output_contracts[name]}
                   for name, symbol in chosen.items()}
        return Graph(nodes=tuple(self.nodes), input_shape=self.input_shape,
                     output_shape=next(iter(outputs.values()))["shape"],
                     output_ref=next(iter(outputs.values()))["ref"],
                     dtype=self.dtype, frontend=self.frontend,
                     input_dtype=next(iter(self.input_contracts.values()))["dtype"],
                     named_inputs=self.input_contracts, named_outputs=outputs)


class OperatorNamespace:
    """Attribute factories for symbolic operator calls.

    ``registry.ops`` is bound to that registry: each attribute is the exact
    declaration its alias names there. The global ``hndl.ops`` is unbound:
    inside a capture it resolves each alias against the registry the capture
    runs under, the one passed to ``resolve_callable`` or
    ``network_from_callable``, so operators registered on that registry need
    no separate namespace. Outside a capture it checks aliases against the
    built-in catalog, and calling a factory there fails with ``E_CAPTURE``.
    """

    def __init__(self, registry=None):
        self._registry = registry

    def __repr__(self):
        return "hndl.ops" if self._registry is None else f"<operators of {self._registry!r}>"

    def __getattr__(self, alias):
        if alias.startswith("_"):
            raise AttributeError(alias)
        if self._registry is not None:
            return self._bound(alias, self._registry.get(alias))
        capture = _ACTIVE.get()
        # Validate the alias now so a typo fails where it is written.
        (Registry.builtins() if capture is None else capture.registry).get(alias)
        return self._late(alias)

    @staticmethod
    def _bound(alias, op):
        def operation(*args, **kwargs):
            capture = _ACTIVE.get()
            if capture is None:
                raise HNDLError("E_CAPTURE", "Symbolic operations require an active authoring capture")
            return capture.emit(op, args, kwargs)

        operation.__name__ = alias
        return operation

    @staticmethod
    def _late(alias):
        def operation(*args, **kwargs):
            capture = _ACTIVE.get()
            if capture is None:
                raise HNDLError("E_CAPTURE", "Symbolic operations require an active authoring capture")
            # Resolved on every call, so a factory read in one capture and
            # called in another follows the registry it is called under.
            return capture.emit(capture.registry.get(alias), args, kwargs)

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
        # A plain contract keeps the single positional symbol; named contracts
        # hand every input to its own keyword parameter.
        result = fn(**capture.inputs) if capture.named_inputs else fn(capture.input)
        if capture.named_outputs:
            return capture.finish(result)
        return capture.finish(_DEFAULT if result is None else result)


def resolve_callable(fn, *, input_shape, output_shape, dtype="float32", registry=None, limits=None,
                     input_dtype=None):
    from .resolver import resolve_graph

    registry = Registry.builtins() if registry is None else registry
    graph = capture_callable(fn, input_shape=input_shape, output_shape=output_shape,
                             dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return resolve_graph(graph, registry=registry, limits=limits)
