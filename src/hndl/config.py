"""Bounded declarative configuration loading without Python execution."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from .capture import Capture, Symbol, _DEFAULT, _error
from .errors import HNDLError
from .registry import Registry
from .settings import SCHEME_NAMES, initializer


MAX_SOURCE_BYTES = 65_536
MAX_LINES = 4096
MAX_PROTOCOL_BYTES = 2_097_152
PARSER_TIMEOUT_SECONDS = 5
_WORKER = Path(__file__).with_name("_parser_worker.py")


def _source_bytes(source):
    if type(source) is not str:
        raise TypeError("Declarative source must be a string; use resolve_callable or resolve_file explicitly")
    if len(source) > MAX_SOURCE_BYTES:
        raise HNDLError("E_RESOURCE", "Source exceeds 65536 UTF-8 bytes")
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError:
        raise HNDLError("E_SYNTAX", "Source must be valid UTF-8 text") from None
    if len(encoded) > MAX_SOURCE_BYTES or len(source.split("\n")) > MAX_LINES:
        raise HNDLError("E_RESOURCE", "Source exceeds byte/line limits")
    return encoded


def _validate_response_program(program):
    """Validate the JSON protocol before using worker-produced records."""
    if type(program) is not dict or set(program) != {"version", "statements"} or type(program["version"]) is not int or program["version"] != 1:
        raise HNDLError("E_RESOURCE", "Invalid parser response format")
    if type(program["statements"]) is not list or len(program["statements"]) > 4096:
        raise HNDLError("E_RESOURCE", "Invalid parser statement list")
    budget = [32_768]

    def fail():
        raise HNDLError("E_RESOURCE", "Invalid parser response format")

    def location(value):
        if type(value) is not dict or set(value) != {"line", "column"}:
            fail()
        if not all(type(item) is int and 0 < item <= MAX_SOURCE_BYTES + 1 for item in value.values()):
            fail()

    def expression(value, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 100 or type(value) is not dict:
            fail()
        kind = value.get("kind")
        if kind == "literal":
            if set(value) != {"kind", "value"} or type(value["value"]) not in (str, int, float, bool, type(None)):
                fail()
            item = value["value"]
            if type(item) is float and not math.isfinite(item):
                fail()
            if type(item) is int and item.bit_length() > 256:
                fail()
            if type(item) is str and len(item.encode("utf-8")) > 16_384:
                fail()
        elif kind == "name":
            if set(value) != {"kind", "id", "source"} or type(value["id"]) is not str:
                fail()
            location(value["source"])
        elif kind == "call":
            if set(value) != {"kind", "alias", "args", "kwargs", "source"} or type(value["alias"]) is not str:
                fail()
            if type(value["args"]) is not list or type(value["kwargs"]) is not list:
                fail()
            location(value["source"])
            for item in value["args"]:
                expression(item, depth + 1)
            for pair in value["kwargs"]:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
                    fail()
                expression(pair[1], depth + 1)
        elif kind in ("tuple", "list", "tensor_tuple"):
            if set(value) != {"kind", "items"} or type(value["items"]) is not list:
                fail()
            for item in value["items"]:
                expression(item, depth + 1)
        elif kind == "dict":
            if set(value) != {"kind", "items"} or type(value["items"]) is not list:
                fail()
            for pair in value["items"]:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
                    fail()
                expression(pair[1], depth + 1)
        elif kind == "init_call":
            if set(value) != {"kind", "name", "kwargs", "source"} or type(value["name"]) is not str:
                fail()
            if value["name"] not in SCHEME_NAMES or type(value["kwargs"]) is not list:
                fail()
            location(value["source"])
            names = set()
            for pair in value["kwargs"]:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in names:
                    fail()
                names.add(pair[0])
                expression(pair[1], depth + 1)
        else:
            fail()

    for statement in program["statements"]:
        if type(statement) is not dict:
            fail()
        if statement.get("kind") == "expr":
            if set(statement) != {"kind", "value", "source"}:
                fail()
        elif statement.get("kind") == "assign":
            if set(statement) != {"kind", "value", "source", "targets", "unpack"}:
                fail()
            if type(statement["targets"]) is not list or not statement["targets"] or not all(type(name) is str for name in statement["targets"]):
                fail()
            if type(statement["unpack"]) is not bool:
                fail()
        else:
            fail()
        location(statement["source"])
        expression(statement["value"])
    return program


def _parse(source, aliases):
    _source_bytes(source)
    if sys.platform != "linux":
        raise HNDLError("E_RESOURCE", "Declarative parsing currently requires Linux process/resource isolation; no unsafe fallback is available")
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13), (3, 14)):
        raise HNDLError("E_SYNTAX", "Declarative parser supports Python runtimes 3.11 through 3.14")
    aliases = tuple(aliases)
    if "x" in aliases or "out" in aliases:
        raise HNDLError("E_NAME", "Operator aliases x and out are reserved by the declarative frontend")
    shadowed = sorted(SCHEME_NAMES.intersection(aliases))
    if shadowed:
        raise HNDLError("E_NAME", "Operator aliases shadow initializer schemes reserved by the "
                                  f"declarative frontend: {', '.join(shadowed)}")
    payload = json.dumps({"source": source, "aliases": aliases}, ensure_ascii=True).encode("ascii")
    if len(payload) > MAX_PROTOCOL_BYTES:
        raise HNDLError("E_RESOURCE", "Parser request exceeds protocol limit")
    # Files, rather than communicate() pipes, let RLIMIT_FSIZE bound worker
    # output before it reaches parent memory. -I -S excludes site hooks and
    # environment/PYTHONPATH imports; the worker imports only the stdlib.
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        stdin.write(payload)
        stdin.seek(0)
        try:
            result = subprocess.run([sys.executable, "-I", "-S", str(_WORKER)],
                                    stdin=stdin, stdout=stdout, stderr=stderr,
                                    timeout=PARSER_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            raise HNDLError("E_RESOURCE", "Isolated parser exceeded wall-time limit") from None
        except OSError as exc:
            raise HNDLError("E_RESOURCE", f"Cannot start isolated parser: {exc.strerror}") from None
        if result.returncode != 0:
            raise HNDLError("E_RESOURCE", "Isolated parser exited before producing a validated result")
        stdout.seek(0)
        data = stdout.read(MAX_PROTOCOL_BYTES + 1)
    if len(data) > MAX_PROTOCOL_BYTES:
        raise HNDLError("E_RESOURCE", "Isolated parser output exceeded protocol limit")
    def unique_fields(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate protocol field")
            obj[key] = value
        return obj

    def reject_constant(value):
        raise ValueError("nonfinite protocol value")

    try:
        result = json.loads(data, object_pairs_hook=unique_fields, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise HNDLError("E_RESOURCE", "Invalid isolated parser response") from None
    if type(result) is not dict or type(result.get("ok")) is not bool:
        raise HNDLError("E_RESOURCE", "Invalid isolated parser response")
    if set(result) != ({"ok", "program"} if result["ok"] else {"ok", "error"}):
        raise HNDLError("E_RESOURCE", "Invalid isolated parser response fields")
    if not result["ok"]:
        error = result.get("error")
        if type(error) is not dict or error.get("code") not in ("E_SYNTAX", "E_RESOURCE", "E_NAME") or type(error.get("message")) is not str:
            raise HNDLError("E_RESOURCE", "Invalid isolated parser error response")
        if any(key in error and (type(error[key]) is not int or error[key] < 1) for key in ("line", "column")):
            raise HNDLError("E_RESOURCE", "Invalid isolated parser error location")
        raise _error(error["code"], error["message"], error)
    return _validate_response_program(result.get("program"))


def _interpret(program, capture):
    # Every declared external input is prebound as a local tensor variable.
    locals_ = dict(capture.inputs)

    def evaluate(expression):
        kind = expression["kind"]
        if kind == "name":
            name = expression["id"]
            if name not in locals_:
                raise _error("E_NAME", f"Undefined tensor variable {name!r}", expression["source"])
            return locals_[name]
        if kind == "literal":
            return expression["value"]
        if kind in ("tuple", "list", "tensor_tuple"):
            values = [evaluate(item) for item in expression["items"]]
            if kind == "tensor_tuple" and not all(isinstance(item, Symbol) for item in values):
                raise HNDLError("E_OUTPUT_ARITY", "A tensor tuple must contain only tensor symbols")
            return values if kind == "list" else tuple(values)
        if kind == "dict":
            return {key: evaluate(value) for key, value in expression["items"]}
        if kind == "init_call":
            # A scheme record is plain data: no registry lookup, no emitted node.
            arguments = {name: evaluate(value) for name, value in expression["kwargs"]}
            try:
                return initializer(expression["name"], arguments)
            except HNDLError as exc:
                raise _error(exc.code, exc.message, expression["source"]) from None
        # All nested arguments run in Python order before current is bound.
        op = capture.registry.get(expression["alias"])
        args = tuple(evaluate(item) for item in expression["args"])
        kwargs = {name: evaluate(value) for name, value in expression["kwargs"]}
        return capture.emit(op, args, kwargs, source=expression["source"])

    for statement in program["statements"]:
        value = evaluate(statement["value"])
        if statement["kind"] == "assign":
            names = statement["targets"]
            if statement["unpack"]:
                if type(value) is not tuple or len(names) != len(value) or not all(isinstance(item, Symbol) for item in value):
                    raise _error("E_OUTPUT_ARITY", "Tuple unpacking must match the declared tensor outputs", statement["source"])
                locals_.update(zip(names, value))
            else:
                locals_[names[0]] = value
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
