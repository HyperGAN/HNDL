"""Torch-free symbolic graph capture for trusted authoring functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import re
import sys
from typing import TYPE_CHECKING, Any

from .errors import HNDLError, source_location
from .registry import Registry, normalize_arguments
from .resolver import _limits, _validate_port_reservations
from .settings import normalize_settings
from .types import (EXTERNAL_INPUT, EXTERNAL_OUTPUT, Graph, Node, named_contracts,
                    named_dtypes)

if TYPE_CHECKING:
    from .operator import Operator
    from .types import ResolvedPlan


_ACTIVE: ContextVar[Capture | None] = ContextVar("hndl_capture", default=None)
_SCOPE_PREFIX: ContextVar[str] = ContextVar("hndl_scope_prefix", default="")
_DEFAULT = object()
_NODE_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@contextmanager
def scope(prefix: str):
    """Context manager scoping node names in symbolic graph capture."""
    if not isinstance(prefix, str) or not _NODE_ID.fullmatch(prefix):
        raise HNDLError("E_NAME", "Scope prefix must match [a-z][a-z0-9_]*")
    curr = _SCOPE_PREFIX.get()
    new_prefix = f"{curr}_{prefix}" if curr else prefix
    token = _SCOPE_PREFIX.set(new_prefix)
    try:
        yield
    finally:
        _SCOPE_PREFIX.reset(token)


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


def _error(code, message, source=None, node=None):
    return HNDLError(code, message, node=node, **source_location(source))


def _rethrow_with_source(exc, source, node_context):
    if source is not None and exc.line is None:
        raise _error(exc.code, exc.message, source, node=exc.node or node_context) from None
    raise HNDLError(exc.code, exc.message, node=exc.node or node_context, **source_location(source)) from None


def _foreign_operator(registry, op):
    if op.key in registry._identities:
        problem = "declares that identity with a different implementation"
    else:
        problem = "does not hold it"
    hint = "pass the registry that declares it to resolve_callable or network_from_callable"
    local = registry._aliases.get(op.alias)
    if local is not None and local.key == op.key:
        hint += f", or call ops.{op.alias} to use the capture's own binding"
    elif local is not None:
        hint += f" (ops.{op.alias} in this capture binds {local.key}, a different operator)"
    return (f"{op.alias} was taken from another registry's ops and binds {op.key}, but this "
            f"capture's registry {problem}; {hint}")


class Capture:
    """One scoped authoring transaction, shared by the two frontends."""

    def __init__(
        self,
        *,
        input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
        output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
        dtype: str = "float32",
        registry: Registry | None = None,
        frontend: str = "python_callable@1",
        limits: Mapping[str, int] | None = None,
        input_dtype: str | Mapping[str, str] | None = None,
    ) -> None:
        self.registry = Registry.builtins() if registry is None else registry
        shapes, self.named_inputs = named_contracts(input_shape, "input_shape", EXTERNAL_INPUT)
        dtypes = named_dtypes(input_dtype, tuple(shapes))
        self.input_contracts = {name: {"shape": shape, "dtype": dtype if dtypes[name] is None else dtypes[name]}
                                for name, shape in shapes.items()}
        self.output_contracts, self.named_outputs = named_contracts(output_shape, "output_shape", EXTERNAL_OUTPUT)
        _validate_port_reservations(set(self.input_contracts) | set(self.output_contracts), self.registry)
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

    def __enter__(self) -> Capture:
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        _ACTIVE.reset(self._token)
        self._token = None

    def _symbol(self, value, source=None, node=None):
        if not isinstance(value, Symbol) or value._owner is not self:
            raise _error("E_CAPTURE", "Input must be a symbolic tensor from this capture", source, node=node)
        return value

    def _bind_inputs(
        self,
        op: Operator,
        args: list[Any],
        kwargs: dict[str, Any],
        source: Mapping[str, Any] | None,
        node_context: str | None,
    ) -> dict[str, Symbol]:
        ports = tuple(op.input_ports)
        bindings = {}

        if op.variadic is not None:
            prefix = op.variadic
            count = 0
            while args and isinstance(args[0], Symbol):
                bindings[f"{prefix}{count}"] = self._symbol(args.pop(0), source, node=node_context)
                count += 1
            for key in tuple(kwargs):
                if op._variadic_member(prefix, key):
                    if key in bindings:
                        raise _error("E_BINDING", f"Duplicate input binding {key}", source, node=node_context)
                    bindings[key] = self._symbol(kwargs.pop(key), source, node=node_context)
            if len(bindings) < 2 or set(bindings) != {f"{prefix}{i}" for i in range(len(bindings))}:
                raise _error("E_BINDING", f"Variadic inputs require contiguous {prefix}0...{prefix}N bindings with at least two tensors", source, node=node_context)
            if "input_count" in kwargs and (type(kwargs["input_count"]) is not int
                                             or kwargs["input_count"] != len(bindings)):
                raise _error("E_BINDING", "input_count does not match supplied tensors", source, node=node_context)
            kwargs["input_count"] = len(bindings)
        elif len(ports) == 1:
            port = ports[0]
            if args and isinstance(args[0], Symbol):
                bindings[port] = self._symbol(args.pop(0), source, node=node_context)
            if port in kwargs:
                if port in bindings:
                    raise _error("E_BINDING", f"Duplicate input binding {port}", source, node=node_context)
                bindings[port] = self._symbol(kwargs.pop(port), source, node=node_context)
            if port not in bindings:
                if self.current is None:
                    raise _error("E_CURRENT", "No current tensor; supply an explicit input after a multi-output operation", source, node=node_context)
                bindings[port] = self.current
        else:
            for port in ports:
                if args and isinstance(args[0], Symbol):
                    bindings[port] = self._symbol(args.pop(0), source, node=node_context)
                if port in kwargs:
                    if port in bindings:
                        raise _error("E_BINDING", f"Duplicate input binding {port}", source, node=node_context)
                    bindings[port] = self._symbol(kwargs.pop(port), source, node=node_context)
                if port not in bindings:
                    raise _error("E_BINDING", f"Multi-input operation requires explicit port {port}", source, node=node_context)

        return bindings

    def emit(
        self,
        op: Operator,
        positional: Sequence[Any],
        keyword: Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = None,
    ) -> Symbol | tuple[Symbol, ...]:
        # Factories carry exact identities, never whichever alias is newest.
        # ``registry.ops`` binds a declaration when the attribute is read; it
        # may run in any capture whose registry holds that very declaration,
        # as every registry built from ``Registry.builtins()`` does for the
        # built-ins, and nowhere else.
        args = list(positional)
        kwargs = dict(keyword)
        name = kwargs.pop("name", _DEFAULT)
        tie = kwargs.pop("tie", _DEFAULT)
        if tie is not _DEFAULT:
            if not isinstance(tie, Mapping):
                raise _error("E_SCHEMA", "tie must be a mapping of target parameter to source parameter path", source)
            for tp, sp in tie.items():
                if not isinstance(tp, str) or not isinstance(sp, str) or "." not in sp:
                    raise _error("E_SCHEMA", f"Invalid tie reference {sp!r} for {tp!r}; expected 'source_node.param'", source)

        curr_scope = _SCOPE_PREFIX.get()
        node_context = (
            f"{curr_scope}_{name}" if curr_scope else name
        ) if (name is not _DEFAULT and isinstance(name, str) and _NODE_ID.fullmatch(name)) else None

        if self.registry._identities.get(op.key) is not op:
            raise _error("E_CAPTURE", _foreign_operator(self.registry, op), source, node=node_context)
        policy = kwargs.pop("policy", _DEFAULT)
        construction = {key: kwargs.pop(key) for key in ("init", "trainable") if key in kwargs}
        try:
            initialization, trainability = normalize_settings(**construction)
        except HNDLError as exc:
            _rethrow_with_source(exc, source, node_context)
        if policy is not _DEFAULT and type(policy) is not str:
            raise _error("E_ARGUMENT", "policy must be a registered policy name", source, node=node_context)
        policy = None if policy is _DEFAULT else policy
        bindings = self._bind_inputs(op, args, kwargs, source, node_context)

        if name is _DEFAULT:
            name = f"{curr_scope}_n{len(self.nodes)}" if curr_scope else f"n{len(self.nodes)}"
        else:
            if type(name) is not str or _NODE_ID.fullmatch(name) is None:
                raise _error("E_NAME", "Node names must match [a-z][a-z0-9_]*", source)
            if curr_scope:
                name = f"{curr_scope}_{name}"
        if type(name) is not str or _NODE_ID.fullmatch(name) is None:
            raise _error("E_NAME", "Node names must match [a-z][a-z0-9_]*", source)
        if name in self._ids:
            raise _error("E_NAME", f"Duplicate node name {name!r}", source, node=name)
        if len(self.nodes) >= self.max_nodes:
            raise _error("E_RESOURCE", "Graph exceeds max_nodes", source, node=name)

        explicit = set(kwargs)
        if args:
            explicit.update((op.positional_rest,) if op.positional_rest is not None else op.positional_names[:len(args)])
        try:
            normalized = normalize_arguments(op, tuple(args), kwargs, policy=policy)
        except HNDLError as exc:
            _rethrow_with_source(exc, source, node_context)
        profile = op.policies[policy] if policy else None
        policy_fields = set(profile.requires) if profile else set()
        metadata = dict(source or {})
        metadata["argument_origins"] = {
            key: "explicit" if key in explicit else "policy-selected" if key in policy_fields else "operator default"
            for key in normalized
        }
        if profile:
            metadata["policy"] = profile.identity
        if tie is not _DEFAULT:
            metadata["tie"] = dict(tie)
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

    def finish(self, selected: Any = _DEFAULT) -> Graph:
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


def _map_arg_type(arg_type: Any) -> Any:
    from .operator import INTS, PAIR, STRS

    if arg_type is int:
        return int
    if arg_type is float:
        return float
    if arg_type is bool:
        return bool
    if arg_type is str:
        return str
    if arg_type is PAIR:
        return int | tuple[int, int]
    if arg_type is INTS:
        return int | Sequence[int]
    if arg_type is STRS:
        return str | Sequence[str]
    return Any


def build_operator_signature(op: Operator) -> inspect.Signature:
    params: list[inspect.Parameter] = []
    has_required_pos_arg = any(
        a.positional and not a.has_default and not a.inferable for a in op.args.values()
    )

    if len(op.inputs) == 1 and not op.inputs[0].variadic and op.inputs[0].name == "x":
        def_val = inspect.Parameter.empty if has_required_pos_arg else None
        ann = Symbol if has_required_pos_arg else (Symbol | None)
        params.append(
            inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=def_val, annotation=ann)
        )
    elif op.inputs and op.inputs[0].variadic:
        params.append(
            inspect.Parameter("x", inspect.Parameter.VAR_POSITIONAL, annotation=Symbol)
        )
    else:
        for port in op.inputs:
            params.append(
                inspect.Parameter(port.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Symbol)
            )

    seen_default = False if has_required_pos_arg else True
    for name, arg in op.args.items():
        type_ann = _map_arg_type(arg.type)
        if arg.has_default:
            default_val = arg.default
            seen_default = True
        elif arg.inferable:
            default_val = None
            type_ann = type_ann | None
            seen_default = True
        else:
            default_val = inspect.Parameter.empty

        if arg.positional and not seen_default:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        elif arg.positional and seen_default and default_val is not inspect.Parameter.empty:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        else:
            kind = inspect.Parameter.KEYWORD_ONLY

        params.append(inspect.Parameter(name, kind, default=default_val, annotation=type_ann))

    params.append(inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=str | None))
    params.append(inspect.Parameter("policy", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=str | None))
    params.append(inspect.Parameter("init", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Mapping[str, Any] | None))
    params.append(inspect.Parameter("trainable", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Mapping[str, bool] | bool | None))
    params.append(inspect.Parameter("tie", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Mapping[str, str] | None))

    ret_type = (
        tuple[Symbol, ...]
        if (len(op.outputs) > 1 or op.outputs[0].variadic or op.outputs_from is not None)
        else Symbol
    )
    return inspect.Signature(parameters=params, return_annotation=ret_type)


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

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry

    scope = staticmethod(scope)

    def __repr__(self) -> str:
        return "hndl.ops" if self._registry is None else f"<operators of {self._registry!r}>"

    def __dir__(self) -> list[str]:
        if self._registry is not None:
            return sorted(self._registry._aliases.keys())
        capture = _ACTIVE.get()
        registry = Registry.builtins() if capture is None else capture.registry
        return sorted(registry._aliases.keys())

    def __getattr__(self, alias: str) -> Callable[..., Any]:
        if alias.startswith("_"):
            raise AttributeError(alias)
        if self._registry is not None:
            return self._bound(alias, self._registry.get(alias))
        capture = _ACTIVE.get()
        # Validate the alias now so a typo fails where it is written.
        (Registry.builtins() if capture is None else capture.registry).get(alias)
        return self._late(alias)

    @staticmethod
    def _bound(alias: str, op: Operator) -> Callable[..., Any]:
        def operation(*args: Any, **kwargs: Any) -> Any:
            capture = _ACTIVE.get()
            if capture is None:
                raise HNDLError("E_CAPTURE", "Symbolic operations require an active authoring capture")
            lineno = sys._getframe(1).f_lineno
            return capture.emit(op, args, kwargs, source={"line": lineno})

        operation.__name__ = alias
        operation.__doc__ = op.summary + ("\n\n" + op.doc.strip() if op.doc else "")
        operation.__signature__ = build_operator_signature(op)
        return operation

    @staticmethod
    def _late(alias: str) -> Callable[..., Any]:
        def operation(*args: Any, **kwargs: Any) -> Any:
            capture = _ACTIVE.get()
            if capture is None:
                raise HNDLError("E_CAPTURE", "Symbolic operations require an active authoring capture")
            op = capture.registry.get(alias)
            lineno = sys._getframe(1).f_lineno
            return capture.emit(op, args, kwargs, source={"line": lineno})

        operation.__name__ = alias
        capture = _ACTIVE.get()
        reg = Registry.builtins() if capture is None else capture.registry
        try:
            op = reg.get(alias)
            operation.__doc__ = op.summary + ("\n\n" + op.doc.strip() if op.doc else "")
            operation.__signature__ = build_operator_signature(op)
        except Exception:
            pass
        return operation


ops: OperatorNamespace = OperatorNamespace()


def capture_callable(
    fn: Callable[..., Any],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    dtype: str = "float32",
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
) -> Graph:
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


def resolve_callable(
    fn: Callable[..., Any],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    dtype: str = "float32",
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
) -> ResolvedPlan:
    from .resolver import resolve_graph

    registry = Registry.builtins() if registry is None else registry
    graph = capture_callable(fn, input_shape=input_shape, output_shape=output_shape,
                             dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return resolve_graph(graph, registry=registry, limits=limits)
