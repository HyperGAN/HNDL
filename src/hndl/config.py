"""Declarative configuration loading without Python execution.

:mod:`hndl._parser` validates the source's AST in process and turns it into
statement records; this module interprets those records against a capture.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import os
from typing import TYPE_CHECKING

from .capture import Capture, Symbol, _DEFAULT, _error
from .errors import HNDLError
from .registry import Registry
from .settings import initializer
from ._parser import MAX_SOURCE_BYTES, parse as _parse

if TYPE_CHECKING:
    from .types import Graph, ResolvedPlan


class _Interpreter:
    """Evaluate validated statement records into one capture.

    Statements run in order against a local namespace in which every declared
    external input is prebound as a tensor variable. Each statement kind has
    one handler in ``STATEMENTS``; a nested body runs through :meth:`run`.

    A loop runs its body once per iteration, so it emits exactly the nodes the
    same statements written out by hand would. Inside a loop, every source
    location gains the ``iterations`` it ran in, and an explicit ``name=``
    gains the iteration indices as a suffix.
    """

    def __init__(self, capture):
        self.capture = capture
        self.locals = dict(capture.inputs)
        self.constants = {}
        self.macros = {}
        self.macro_call_counts = {}
        # One [index, count] per enclosing loop, outermost first.
        self.iterations = []
        # Loop-suffixed node IDs, for a duplicate-name message that explains them.
        self.suffixed = {}
        # Explicit node names passed outside loops.
        self.explicit_names = set()

    def run(self, statements):
        for statement in statements:
            self.STATEMENTS[statement["kind"]](self, statement)

    def expr_statement(self, statement):
        self.evaluate(statement["value"])

    def assign_statement(self, statement):
        value = self.evaluate(statement["value"])
        names = statement["targets"]
        if statement["unpack"]:
            if type(value) is not tuple or len(names) != len(value) or not all(isinstance(item, Symbol) for item in value):
                raise _error("E_OUTPUT_ARITY", "Tuple unpacking must match the declared tensor outputs",
                             self.where(statement["source"]))
            for name, val in zip(names, value):
                if name != "_":
                    self.locals[name] = val
                    if isinstance(val, Symbol) and val.ref.startswith("node:"):
                        nid = val.ref[5:].split("/", 1)[0]
                        node_idx = next((i for i, n in enumerate(self.capture.nodes) if n.id == nid), None)
                        if node_idx is not None and "binding" not in self.capture.nodes[node_idx].source:
                            self.capture.nodes[node_idx] = replace(self.capture.nodes[node_idx],
                                source={**self.capture.nodes[node_idx].source, "binding": name})
        else:
            if names[0] != "_":
                self.locals[names[0]] = value
                if isinstance(value, Symbol) and value.ref.startswith("node:"):
                    nid = value.ref[5:].split("/", 1)[0]
                    node_idx = next((i for i, n in enumerate(self.capture.nodes) if n.id == nid), None)
                    if node_idx is not None and "binding" not in self.capture.nodes[node_idx].source:
                        self.capture.nodes[node_idx] = replace(self.capture.nodes[node_idx],
                            source={**self.capture.nodes[node_idx].source, "binding": names[0]})

    def constant_def_statement(self, statement):
        name = statement["name"]
        if name in self.constants:
            raise _error("E_BINDING", f"Constant '{name}' cannot be rebound", self.where(statement["source"]))
        val = statement["value"]
        value = val["value"] if "value" in val else self.evaluate(val)
        self.constants[name] = value

    def macro_def_statement(self, statement):
        name = statement["name"]
        if name in self.macros:
            raise _error("E_NAME", f"Duplicate macro name {name!r}", self.where(statement["source"]))
        self.macros[name] = statement

    def for_statement(self, statement):
        capture = self.capture
        count = statement["count"]
        # Fail before emitting anything: the parser multiplied out the body.
        if len(capture.nodes) + statement["nodes"] > capture.max_nodes:
            body = statement["nodes"] // (count if count else 1)
            prefix = f"range({_amount(count)})" if statement.get("type") != "tuple" else f"tuple loop of {count} items"
            raise _error("E_RESOURCE", f"{prefix} repeats a body of {_amount(body)} node"
                         f"{'' if body == 1 else 's'}; unrolled, the graph would exceed max_nodes "
                         f"({capture.max_nodes})", self.where(statement["source"]))
        level = [0, count]
        self.iterations.append(level)
        try:
            if statement.get("type") == "tuple":
                target = statement["target"]
                for index, item in enumerate(statement["items"]):
                    level[0] = index
                    self.locals[target] = item
                    self.run(statement["body"])
                self.locals.pop(statement["target"], None)
            else:
                for index in range(count):
                    level[0] = index
                    self.run(statement["body"])
        finally:
            self.iterations.pop()

    STATEMENTS = {
        "expr": expr_statement,
        "assign": assign_statement,
        "for": for_statement,
        "constant_def": constant_def_statement,
        "macro_def": macro_def_statement,
    }

    def where(self, source):
        """``source`` plus the loop iterations it is running in, if any."""
        if not self.iterations:
            return source
        return {**source, "iterations": [list(level) for level in self.iterations]}

    def node_name(self, name, source):
        """Append the iteration suffix to an explicit name inside a loop.

        The suffix is each enclosing loop's index, outermost first, joined by
        underscores: ``block3`` in one loop, ``block1_3`` in two.
        """
        if type(name) is not str:
            return name
        earlier = self.suffixed.get(name)
        if not self.iterations:
            if earlier is not None:
                raise _error("E_NAME", f"Duplicate node name {name!r}: an earlier loop already gave that name to "
                                       f"name={earlier[0]!r} with iteration suffix {earlier[1]!r}", source)
            self.explicit_names.add(name)
            return name
        suffix = "_".join(str(index) for index, _ in self.iterations)
        node_id = name + suffix
        if node_id in self.capture._ids:
            earlier = self.suffixed.get(node_id)
            if earlier is not None:
                owner = f"name={earlier[0]!r} with iteration suffix {earlier[1]!r}"
            elif node_id in self.explicit_names:
                owner = f"an explicit name={node_id!r}"
            else:
                owner = f"an auto-generated ID {node_id!r}"
            raise _error("E_NAME", f"Duplicate node name {node_id!r}: name={name!r} in a loop gets the iteration "
                                   f"suffix {suffix!r}, but an earlier node already has that ID ({owner})", source)
        self.suffixed[node_id] = (name, suffix)
        return node_id

    def evaluate(self, expression):
        kind = expression["kind"]
        if kind == "constant":
            return expression.get("value", self.constants.get(expression["id"]))
        if kind == "name":
            name = expression["id"]
            if name in self.constants:
                return self.constants[name]
            if name not in self.locals:
                raise _error("E_NAME", f"Undefined tensor variable {name!r}", self.where(expression["source"]))
            return self.locals[name]
        if kind == "literal":
            return expression["value"]
        if kind in ("tuple", "list", "tensor_tuple"):
            values = [self.evaluate(item) for item in expression["items"]]
            if kind == "tensor_tuple" and not all(isinstance(item, Symbol) for item in values):
                raise _error("E_OUTPUT_ARITY", "A tensor tuple must contain only tensor symbols",
                             self.where(expression["source"]))
            return values if kind == "list" else tuple(values)
        if kind == "dict":
            return {key: self.evaluate(value) for key, value in expression["items"]}
        if kind == "init_call":
            # A scheme record is plain data: no registry lookup, no emitted node.
            arguments = {name: self.evaluate(value) for name, value in expression["kwargs"]}
            try:
                return initializer(expression["name"], arguments)
            except HNDLError as exc:
                raise _error(exc.code, exc.message, self.where(expression["source"])) from None
        if kind == "macro_call":
            from .capture import ops
            macro_name = expression["name"]
            macro = self.macros[macro_name]
            call_idx = self.macro_call_counts.get(macro_name, 0)
            self.macro_call_counts[macro_name] = call_idx + 1
            scope_prefix = f"{macro_name}_{call_idx}"

            args = tuple(self.evaluate(item) for item in expression["args"])
            kwargs = {name: self.evaluate(value) for name, value in expression["kwargs"]}
            params = macro["params"]
            if len(args) > len(params):
                raise _error("E_ARGUMENT", f"Macro {macro_name} accepts at most {len(params)} positional arguments, got {len(args)}", self.where(expression["source"]))
            frame = dict(zip(params, args))
            for k, v in kwargs.items():
                if k in frame:
                    raise _error("E_ARGUMENT", f"Macro {macro_name} parameter {k!r} bound twice", self.where(expression["source"]))
                if k not in params:
                    raise _error("E_ARGUMENT", f"Macro {macro_name} has no parameter {k!r}", self.where(expression["source"]))
                frame[k] = v
            missing = [p for p in params if p not in frame]
            if missing:
                raise _error("E_ARGUMENT", f"Macro {macro_name} missing required argument(s): {', '.join(missing)}", self.where(expression["source"]))

            saved_locals = dict(self.locals)
            self.locals.update(frame)
            ret_val = None
            has_return = False
            try:
                with ops.scope(scope_prefix):
                    for stmt in macro["body"]:
                        if stmt["kind"] == "return":
                            has_return = True
                            if stmt["value"] is not None:
                                ret_val = self.evaluate(stmt["value"])
                            else:
                                ret_val = self.capture.current
                            break
                        self.STATEMENTS[stmt["kind"]](self, stmt)
            finally:
                self.locals = saved_locals
            if has_return:
                return ret_val
            return self.capture.current
        # All nested arguments run in Python order before current is bound.
        op = self.capture.registry.get(expression["alias"])
        args = tuple(self.evaluate(item) for item in expression["args"])
        kwargs = {name: self.evaluate(value) for name, value in expression["kwargs"]}
        source = self.where(expression["source"])
        if "name" in kwargs:
            kwargs["name"] = self.node_name(kwargs["name"], source)
        return self.capture.emit(op, args, kwargs, source=source)


def _amount(count):
    # A multiplied-out count can have thousands of digits; say so instead.
    return str(count) if count < 10**15 else "more than 10**15"


def _interpret(program, capture):
    interpreter = _Interpreter(capture)
    interpreter.run(program["statements"])
    locals_ = interpreter.locals
    if capture.named_outputs:
        # Named outputs are selected by binding each declared name as a local.
        selected = {name: locals_[name] for name in capture.output_contracts
                    if name in locals_}
        if "out" in capture.output_contracts and "out" not in selected and capture.current is not None:
            selected["out"] = capture.current
        return capture.finish(selected)
    return capture.finish(locals_.get("out", _DEFAULT))


def capture_config(
    source: str,
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    dtype: str = "float32",
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
) -> Graph:
    registry = Registry.builtins() if registry is None else registry
    program = _parse(source, registry.aliases)
    with Capture(input_shape=input_shape, output_shape=output_shape, dtype=dtype,
                 registry=registry, frontend="python_config@1", limits=limits, input_dtype=input_dtype) as capture:
        return _interpret(program, capture)


def resolve(
    source: str,
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
    graph = capture_config(source, input_shape=input_shape, output_shape=output_shape,
                           dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return resolve_graph(graph, registry=registry, limits=limits)


def resolve_file(
    path: str | os.PathLike[str],
    *,
    input_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    output_shape: tuple[int | str, ...] | Mapping[str, tuple[int | str, ...]],
    dtype: str = "float32",
    registry: Registry | None = None,
    limits: Mapping[str, int] | None = None,
    input_dtype: str | Mapping[str, str] | None = None,
) -> ResolvedPlan:
    with open(path, "rb") as source_file:
        raw = source_file.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise HNDLError("E_RESOURCE", "Configuration file exceeds byte limit")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HNDLError("E_SYNTAX", "Configuration file is not valid UTF-8") from None
    return resolve(source, input_shape=input_shape, output_shape=output_shape,
                   dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
