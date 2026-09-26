"""Declarative configuration loading without Python execution.

:mod:`hndl._parser` validates the source's AST in process and turns it into
statement records; this module interprets those records against a capture.
"""

from __future__ import annotations

from .capture import Capture, Symbol, _DEFAULT, _error
from .errors import HNDLError
from .registry import Registry
from .settings import initializer
from ._parser import MAX_SOURCE_BYTES, parse as _parse


class _Interpreter:
    """Evaluate validated statement records into one capture.

    Statements run in order against a local namespace in which every declared
    external input is prebound as a tensor variable. Each statement kind has
    one handler in ``STATEMENTS``; a nested body runs through :meth:`run`.
    """

    def __init__(self, capture):
        self.capture = capture
        self.locals = dict(capture.inputs)

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
                raise _error("E_OUTPUT_ARITY", "Tuple unpacking must match the declared tensor outputs", statement["source"])
            self.locals.update(zip(names, value))
        else:
            self.locals[names[0]] = value

    STATEMENTS = {"expr": expr_statement, "assign": assign_statement}

    def evaluate(self, expression):
        kind = expression["kind"]
        if kind == "name":
            name = expression["id"]
            if name not in self.locals:
                raise _error("E_NAME", f"Undefined tensor variable {name!r}", expression["source"])
            return self.locals[name]
        if kind == "literal":
            return expression["value"]
        if kind in ("tuple", "list", "tensor_tuple"):
            values = [self.evaluate(item) for item in expression["items"]]
            if kind == "tensor_tuple" and not all(isinstance(item, Symbol) for item in values):
                raise HNDLError("E_OUTPUT_ARITY", "A tensor tuple must contain only tensor symbols")
            return values if kind == "list" else tuple(values)
        if kind == "dict":
            return {key: self.evaluate(value) for key, value in expression["items"]}
        if kind == "init_call":
            # A scheme record is plain data: no registry lookup, no emitted node.
            arguments = {name: self.evaluate(value) for name, value in expression["kwargs"]}
            try:
                return initializer(expression["name"], arguments)
            except HNDLError as exc:
                raise _error(exc.code, exc.message, expression["source"]) from None
        # All nested arguments run in Python order before current is bound.
        op = self.capture.registry.get(expression["alias"])
        args = tuple(self.evaluate(item) for item in expression["args"])
        kwargs = {name: self.evaluate(value) for name, value in expression["kwargs"]}
        return self.capture.emit(op, args, kwargs, source=expression["source"])


def _interpret(program, capture):
    interpreter = _Interpreter(capture)
    interpreter.run(program["statements"])
    locals_ = interpreter.locals
    if capture.named_outputs:
        # Named outputs are selected by binding each declared name as a local.
        return capture.finish({name: locals_[name] for name in capture.output_contracts
                               if name in locals_})
    return capture.finish(locals_.get("out", _DEFAULT))


def capture_config(source, *, input_shape, output_shape, dtype="float32", registry=None, limits=None,
                   input_dtype=None):
    registry = Registry.builtins() if registry is None else registry
    program = _parse(source, registry.aliases)
    with Capture(input_shape=input_shape, output_shape=output_shape, dtype=dtype,
                 registry=registry, frontend="python_config@1", limits=limits, input_dtype=input_dtype) as capture:
        return _interpret(program, capture)


def resolve(source, *, input_shape, output_shape, dtype="float32", registry=None, limits=None, input_dtype=None):
    from .resolver import resolve_graph

    registry = Registry.builtins() if registry is None else registry
    graph = capture_config(source, input_shape=input_shape, output_shape=output_shape,
                           dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return resolve_graph(graph, registry=registry, limits=limits)


def resolve_file(path, *, input_shape, output_shape, dtype="float32", registry=None, limits=None, input_dtype=None):
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
