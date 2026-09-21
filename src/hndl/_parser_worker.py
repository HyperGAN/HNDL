"""Standalone, resource-limited AST validator. Never imports application code.

Invoked by config.py with ``python -I -S <this file>``. The wire format is
bounded JSON; input is always data, never executable Python source.
"""

from __future__ import annotations

import ast
import json
import math
import sys
import textwrap


MAX_SOURCE_BYTES = 65_536
MAX_LINES = 4096
MAX_AST_NODES = 16_384
MAX_DEPTH = 96
MAX_LITERAL_ITEMS = 1024
MAX_LITERAL_BYTES = 16_384
MAX_INTEGER_BITS = 256
MAX_PROTOCOL_BYTES = 2_097_152
MAX_INITIALIZER_KWARGS = 8

# The only calls permitted inside a literal container, and only as a direct
# value of an ``init={...}`` mapping. This fixed list is duplicated from
# hndl.settings on purpose: the worker imports nothing from the package.
INITIALIZER_CALLS = frozenset({"xavier_uniform", "xavier_normal", "kaiming_uniform",
                               "kaiming_normal", "truncated_normal", "normal",
                               "uniform", "orthogonal"})


class Rejected(Exception):
    def __init__(self, code, message, location=None):
        self.code, self.message, self.location = code, message, location or {}


def reject(message, node=None, *, code="E_SYNTAX"):
    location = {} if node is None else {"line": node.lineno, "column": node.col_offset + 1}
    raise Rejected(code, message, location)


def valid_name(name):
    return isinstance(name, str) and name.isidentifier() and not name.startswith("__")


class Validator:
    def __init__(self, aliases, offsets, lines):
        self.aliases = frozenset(aliases)
        self.offsets = offsets
        self.lines = lines

    def location(self, node):
        prefix = self.lines[node.lineno - 1].encode("utf-8")[:node.col_offset].decode("utf-8")
        return {"line": node.lineno,
                "column": len(prefix) + 1 + self.offsets[node.lineno - 1]}

    def initializer_call(self, node):
        """One allowlisted initializer scheme; keyword-only, literal arguments."""
        if node.args:
            reject("Initializer calls take keyword arguments only", node)
        if len(node.keywords) > MAX_INITIALIZER_KWARGS:
            reject("Initializer call exceeds keyword limit", node, code="E_RESOURCE")
        names = [keyword.arg for keyword in node.keywords]
        if None in names or len(set(names)) != len(names):
            reject("Keyword expansion or duplicate keywords are not permitted", node)
        if any(not valid_name(name) for name in names):
            reject("Invalid initializer keyword", node)
        return {"kind": "init_call", "name": node.func.id,
                "kwargs": [[keyword.arg, self.literal(keyword.value)] for keyword in node.keywords],
                "source": self.location(node)}

    def literal(self, node, depth=0, *, initializers=False, call=False):
        if depth > MAX_DEPTH:
            reject("Literal nesting exceeds limit", node, code="E_RESOURCE")
        if isinstance(node, ast.Call):
            # Only an init= mapping value may be a call, and only one of the
            # fixed initializer schemes; operator aliases stay out of literals.
            if not call or not isinstance(node.func, ast.Name) or node.func.id not in INITIALIZER_CALLS:
                reject("Arguments must be literal values or tensor expressions; containers cannot contain calls", node)
            return self.initializer_call(node)
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) not in (int, float, bool, str, type(None)):
                reject("Unsupported literal type", node)
            if type(value) is int and value.bit_length() > MAX_INTEGER_BITS:
                reject("Integer literal exceeds bit limit", node, code="E_RESOURCE")
            if type(value) is float and not math.isfinite(value):
                reject("Numeric literals must be finite", node)
            if type(value) is str and len(value.encode("utf-8")) > MAX_LITERAL_BYTES:
                reject("String literal exceeds limit", node, code="E_RESOURCE")
            return {"kind": "literal", "value": value}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            if not isinstance(node.operand, ast.Constant) or type(node.operand.value) not in (int, float):
                reject("Only a sign on a numeric literal is permitted", node)
            value = self.literal(node.operand)["value"]
            return {"kind": "literal", "value": -value if isinstance(node.op, ast.USub) else value}
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > MAX_LITERAL_ITEMS:
                reject("Literal container exceeds item limit", node, code="E_RESOURCE")
            return {"kind": "tuple" if isinstance(node, ast.Tuple) else "list",
                    "items": [self.literal(item, depth + 1) for item in node.elts]}
        if isinstance(node, ast.Dict):
            if len(node.keys) > MAX_LITERAL_ITEMS:
                reject("Literal dictionary exceeds item limit", node, code="E_RESOURCE")
            pairs = []
            keys = set()
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    reject("Literal dictionaries require unique string keys", node)
                self.literal(key, depth + 1)
                if key.value in keys:
                    reject("Duplicate literal dictionary key", key)
                keys.add(key.value)
                pairs.append([key.value, self.literal(value, depth + 1, call=initializers)])
            return {"kind": "dict", "items": pairs}
        reject("Arguments must be literal values or tensor expressions; containers cannot contain calls", node)

    def expression(self, node, *, assignment=False, initializers=False):
        if isinstance(node, ast.Name):
            if not valid_name(node.id):
                reject("Invalid local name", node)
            return {"kind": "name", "id": node.id, "source": self.location(node)}
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self.aliases:
                reject("Calls must name a registered operator directly", node)
            if not valid_name(node.func.id):
                reject("Invalid operator alias", node)
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                reject("Star argument expansion is not permitted", node)
            names = [keyword.arg for keyword in node.keywords]
            if None in names or len(set(names)) != len(names):
                reject("Keyword expansion or duplicate keywords are not permitted", node)
            return {"kind": "call", "alias": node.func.id,
                    "args": [self.expression(arg) for arg in node.args],
                    "kwargs": [[keyword.arg, self.expression(keyword.value, initializers=keyword.arg == "init")]
                               for keyword in node.keywords],
                    "source": self.location(node)}
        if assignment:
            if isinstance(node, ast.Tuple) and all(isinstance(item, (ast.Name, ast.Call)) for item in node.elts):
                return {"kind": "tensor_tuple", "items": [self.expression(item, assignment=True) for item in node.elts]}
            reject("Assignments must bind tensor expressions, not literal values", node)
        return self.literal(node, initializers=initializers)

    def statements(self, module):
        statements = []
        for statement in module.body:
            if isinstance(statement, ast.Expr):
                if not isinstance(statement.value, ast.Call):
                    reject("Only registered calls may be expression statements", statement)
                statements.append({"kind": "expr", "value": self.expression(statement.value),
                                   "source": self.location(statement)})
            elif isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    reject("Chained assignment is not permitted", statement)
                target = statement.targets[0]
                unpack = isinstance(target, (ast.Tuple, ast.List))
                elements = target.elts if unpack else [target]
                if not elements or not all(isinstance(item, ast.Name) and valid_name(item.id) for item in elements):
                    reject("Assignment targets must be names or a flat tuple/list of names", statement)
                names = [item.id for item in elements]
                if len(set(names)) != len(names):
                    reject("Unpacking targets must be distinct", statement)
                if any(name in self.aliases for name in names):
                    reject("Registered operator aliases cannot be rebound", statement, code="E_NAME")
                statements.append({"kind": "assign", "targets": names, "unpack": unpack,
                                   "value": self.expression(statement.value, assignment=True),
                                   "source": self.location(statement)})
            else:
                reject("Statement is outside the declarative Python subset", statement)
        return statements


def parse_request(request):
    if type(request) is not dict or set(request) != {"source", "aliases"}:
        raise Rejected("E_SYNTAX", "Invalid parser request")
    source, aliases = request["source"], request["aliases"]
    if type(source) is not str or type(aliases) is not list or not all(valid_name(alias) for alias in aliases):
        raise Rejected("E_SYNTAX", "Invalid parser request values")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES or len(source.split("\n")) > MAX_LINES:
        raise Rejected("E_RESOURCE", "Source exceeds configured limits")
    dedented = textwrap.dedent(source)
    original_lines = source.split("\n")
    clean_lines = dedented.split("\n")
    offsets = [max(0, len(original) - len(clean))
               for original, clean in zip(original_lines, clean_lines)]
    try:
        module = ast.parse(dedented, mode="exec", feature_version=(3, 11))
    except SyntaxError as exc:
        line = exc.lineno or 1
        offset = offsets[line - 1] if line <= len(offsets) else 0
        raise Rejected("E_SYNTAX", exc.msg, {"line": line, "column": (exc.offset or 1) + offset}) from None
    pending = [(module, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_DEPTH:
            raise Rejected("E_RESOURCE", "AST exceeds node/depth limits")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    validator = Validator(aliases, offsets, clean_lines)
    try:
        statements = validator.statements(module)
    except Rejected as exc:
        # Validator errors generated through reject use dedented positions.
        line = exc.location.get("line")
        if line and line <= len(offsets):
            byte_column = exc.location["column"] - 1
            prefix = clean_lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8")
            exc.location["column"] = len(prefix) + 1 + offsets[line - 1]
        raise
    return {"version": 1, "statements": statements}


def main():
    if sys.platform != "linux":
        raise RuntimeError("This parser worker requires Linux resource isolation")
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (268_435_456, 268_435_456))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_PROTOCOL_BYTES, MAX_PROTOCOL_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    try:
        data = sys.stdin.buffer.read(MAX_PROTOCOL_BYTES + 1)
        if len(data) > MAX_PROTOCOL_BYTES:
            raise Rejected("E_RESOURCE", "Parser request exceeds limit")
        result = {"ok": True, "program": parse_request(json.loads(data))}
    except Rejected as exc:
        result = {"ok": False, "error": {"code": exc.code, "message": exc.message, **exc.location}}
    except (MemoryError, RecursionError):
        result = {"ok": False, "error": {"code": "E_RESOURCE", "message": "Parser resource limit exceeded"}}
    except (ValueError, UnicodeError, TypeError):
        result = {"ok": False, "error": {"code": "E_SYNTAX", "message": "Malformed parser input"}}
    encoded = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    if len(encoded) > MAX_PROTOCOL_BYTES:
        encoded = b'{"ok":false,"error":{"code":"E_RESOURCE","message":"Parser output exceeds limit"}}'
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
