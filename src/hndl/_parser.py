"""Validate declarative config source into plain statement records.

Source is parsed with :func:`ast.parse` and is never compiled or executed. The
whole tree is checked against an allowlist before anything is looked up in a
registry: statements are expression calls and assignments, calls name a
registered operator alias directly, and arguments are literals, tensor names
or nested calls. There are no imports, attribute access, subscripts, operators
or comprehensions.

Each accepted statement becomes a small record (``{"kind": "expr" | "assign",
...}``) that :mod:`hndl.config` interprets. A new statement type is one handler
in ``Validator.STATEMENTS`` here and one in ``config._Interpreter.STATEMENTS``.
"""

from __future__ import annotations

import ast
import io
import keyword
import math
import textwrap
import tokenize
import warnings

from .errors import HNDLError
from .settings import SCHEME_NAMES


MAX_SOURCE_BYTES = 65_536
# Parsing happens in the caller's thread. On CPython 3.11-3.13, ``ast.parse``
# of a few thousand chained operators (``-------1``, ``relu()()()...``,
# ``lambda: lambda: ...``) exhausts a small thread stack, 512 KiB for example,
# and the process dies with SIGSEGV instead of raising. Every level of AST
# nesting costs a bracket or an operator/keyword token, so ``_screen`` bounds
# both on the token stream, which the tokenizer produces iteratively, before
# ``ast.parse`` sees the source. Valid configs contain no counted token and
# rarely nest brackets past five. Within these bounds, every adversarial
# source tried at the size cap parses or fails with HNDLError in a 256 KiB
# thread on 3.11-3.14 (tests/test_config.py keeps a set of them).
MAX_DEPTH = 50
MAX_OPERATORS = 32
# The recursive validator's own backstop, in AST levels.
MAX_AST_DEPTH = 3 * MAX_DEPTH

_OPENERS, _CLOSERS = "([{", ")]}"
_SEPARATORS = frozenset({",", ";", "=", ":"})
# Keywords the grammar accepts. A statement keyword added to the grammar
# (``for`` and ``in`` for loops) belongs here too.
_ACCEPTED_KEYWORDS = frozenset({"True", "False", "None"})
_OPERANDS = (tokenize.NAME, tokenize.NUMBER, tokenize.STRING)
_LAYOUT = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)


def _valid_name(name):
    return isinstance(name, str) and name.isidentifier() and not name.startswith("__")


def _source_text(source):
    if type(source) is not str:
        raise TypeError("Declarative source must be a string; use resolve_callable or resolve_file explicitly")
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError:
        raise HNDLError("E_SYNTAX", "Source must be valid UTF-8 text") from None
    if len(encoded) > MAX_SOURCE_BYTES:
        raise HNDLError("E_RESOURCE", f"Source exceeds {MAX_SOURCE_BYTES} UTF-8 bytes")
    return source


def _counts(token, previous, following):
    """Whether ``token`` adds a level of nesting that is not a bracket."""
    if token.type == tokenize.NAME:
        return keyword.iskeyword(token.string) and token.string not in _ACCEPTED_KEYWORDS
    if token.type != tokenize.OP or token.string in _CLOSERS or token.string in _SEPARATORS:
        return False
    after_operand = previous is not None and (previous.type in _OPERANDS or previous.string in _CLOSERS)
    if token.string in _OPENERS:
        # A call on a name is the grammar's own; a bracket right after a
        # literal or a closing bracket calls or subscripts a result, and a
        # chain of those nests one level each.
        return previous is not None and (previous.type in (tokenize.NUMBER, tokenize.STRING)
                                         or previous.string in _CLOSERS)
    if token.string in "+-" and following is not None and following.type == tokenize.NUMBER:
        # A sign on a number is part of the literal, unless it follows an
        # operand (binary) or another operator (a chain of signs).
        return after_operand or (previous is not None and previous.type == tokenize.OP
                                 and previous.string not in _OPENERS and previous.string not in _SEPARATORS)
    return True


def _screen(source, offsets):
    """Bound bracket depth and operator/keyword count before ``ast.parse``.

    A tokenizer error ends the screen early: ``ast.parse`` stops at the same
    token and reports it, so the unscreened rest is never parsed.
    """
    def fail(code, message, token):
        line, column = token.start
        offset = offsets[line - 1] if line <= len(offsets) else 0
        raise HNDLError(code, message, line=line, column=column + 1 + offset)

    depth = operators = 0
    previous = None
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        following = next(tokens, None)
        while following is not None:
            token, following = following, next(tokens, None)
            if _counts(token, previous, following):
                operators += 1
                if operators > MAX_OPERATORS:
                    fail("E_SYNTAX", f"Source uses more than {MAX_OPERATORS} Python operators or keywords; "
                                     "the declarative subset accepts none of them", token)
            if token.type == tokenize.OP and token.string in _OPENERS:
                depth += 1
                if depth > MAX_DEPTH:
                    fail("E_RESOURCE", f"Source nests brackets more than {MAX_DEPTH} levels deep", token)
            elif token.type == tokenize.OP and token.string in _CLOSERS:
                depth = max(0, depth - 1)
            if token.type not in _LAYOUT:
                previous = token
    except HNDLError:  # a ValueError, like the tokenizer's null-byte error
        raise
    except (tokenize.TokenError, SyntaxError, ValueError):  # SyntaxError covers IndentationError
        return


class Validator:
    """Translate one parsed module into statement records, or raise HNDLError."""

    def __init__(self, aliases, offsets, lines):
        self.aliases = frozenset(aliases)
        self.offsets = offsets
        self.lines = lines

    def location(self, node):
        # col_offset counts UTF-8 bytes of the dedented line; report characters
        # of the original line.
        prefix = self.lines[node.lineno - 1].encode("utf-8")[:node.col_offset].decode("utf-8", "ignore")
        return {"line": node.lineno, "column": len(prefix) + 1 + self.offsets[node.lineno - 1]}

    def reject(self, message, node, code="E_SYNTAX"):
        location = self.location(node)
        raise HNDLError(code, message, line=location["line"], column=location["column"])

    # Statements -----------------------------------------------------------

    def statements(self, body):
        return [self.statement(node) for node in body]

    def statement(self, node):
        handler = self.STATEMENTS.get(type(node))
        if handler is None:
            self.reject("Statement is outside the declarative Python subset", node)
        return handler(self, node)

    def expr_statement(self, node):
        if not isinstance(node.value, ast.Call):
            self.reject("Only registered calls may be expression statements", node)
        return {"kind": "expr", "value": self.expression(node.value), "source": self.location(node)}

    def assign_statement(self, node):
        if len(node.targets) != 1:
            self.reject("Chained assignment is not permitted", node)
        target = node.targets[0]
        unpack = isinstance(target, (ast.Tuple, ast.List))
        elements = target.elts if unpack else [target]
        if not elements or not all(isinstance(item, ast.Name) and _valid_name(item.id) for item in elements):
            self.reject("Assignment targets must be names or a flat tuple/list of names", node)
        names = [item.id for item in elements]
        if len(set(names)) != len(names):
            self.reject("Unpacking targets must be distinct", node)
        if any(name in self.aliases for name in names):
            self.reject("Registered operator aliases cannot be rebound", node, code="E_NAME")
        return {"kind": "assign", "targets": names, "unpack": unpack,
                "value": self.expression(node.value, assignment=True), "source": self.location(node)}

    STATEMENTS = {ast.Expr: expr_statement, ast.Assign: assign_statement}

    # Expressions ----------------------------------------------------------

    def keywords(self, node):
        names = [keyword.arg for keyword in node.keywords]
        if None in names or len(set(names)) != len(names):
            self.reject("Keyword expansion or duplicate keywords are not permitted", node)
        return names

    def expression(self, node, *, assignment=False, initializers=False):
        if isinstance(node, ast.Name):
            if not _valid_name(node.id):
                self.reject("Invalid local name", node)
            return {"kind": "name", "id": node.id, "source": self.location(node)}
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self.aliases:
                self.reject("Calls must name a registered operator directly", node)
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                self.reject("Star argument expansion is not permitted", node)
            self.keywords(node)
            return {"kind": "call", "alias": node.func.id,
                    "args": [self.expression(arg) for arg in node.args],
                    "kwargs": [[keyword.arg, self.expression(keyword.value, initializers=keyword.arg == "init")]
                               for keyword in node.keywords],
                    "source": self.location(node)}
        if assignment:
            if isinstance(node, ast.Tuple) and all(isinstance(item, (ast.Name, ast.Call)) for item in node.elts):
                return {"kind": "tensor_tuple", "items": [self.expression(item, assignment=True) for item in node.elts]}
            self.reject("Assignments must bind tensor expressions, not literal values", node)
        return self.literal(node, initializers=initializers)

    def initializer_call(self, node):
        """One allowlisted initializer scheme; keyword-only, literal arguments."""
        if node.args:
            self.reject("Initializer calls take keyword arguments only", node)
        if any(not _valid_name(name) for name in self.keywords(node)):
            self.reject("Invalid initializer keyword", node)
        return {"kind": "init_call", "name": node.func.id,
                "kwargs": [[keyword.arg, self.literal(keyword.value)] for keyword in node.keywords],
                "source": self.location(node)}

    def literal(self, node, *, initializers=False, call=False):
        if isinstance(node, ast.Call):
            # Only an init= mapping value may be a call, and only one of the
            # fixed initializer schemes; operator aliases stay out of literals.
            if not call or not isinstance(node.func, ast.Name) or node.func.id not in SCHEME_NAMES:
                self.reject("Arguments must be literal values or tensor expressions; containers cannot contain calls",
                            node)
            return self.initializer_call(node)
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) not in (int, float, bool, str, type(None)):
                self.reject("Unsupported literal type", node)
            if type(value) is float and not math.isfinite(value):
                self.reject("Numeric literals must be finite", node)
            return {"kind": "literal", "value": value}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            if not isinstance(node.operand, ast.Constant) or type(node.operand.value) not in (int, float):
                self.reject("Only a sign on a numeric literal is permitted", node)
            value = self.literal(node.operand)["value"]
            return {"kind": "literal", "value": -value if isinstance(node.op, ast.USub) else value}
        if isinstance(node, (ast.List, ast.Tuple)):
            return {"kind": "tuple" if isinstance(node, ast.Tuple) else "list",
                    "items": [self.literal(item) for item in node.elts]}
        if isinstance(node, ast.Dict):
            pairs = []
            keys = set()
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    self.reject("Literal dictionaries require unique string keys", node)
                if key.value in keys:
                    self.reject("Duplicate literal dictionary key", key)
                keys.add(key.value)
                pairs.append([key.value, self.literal(value, call=initializers)])
            return {"kind": "dict", "items": pairs}
        self.reject("Arguments must be literal values or tensor expressions; containers cannot contain calls", node)


def parse(source, aliases):
    """Validate ``source`` against the allowlist and return its statement records."""
    source = _source_text(source)
    aliases = tuple(aliases)
    if "x" in aliases or "out" in aliases:
        raise HNDLError("E_NAME", "Operator aliases x and out are reserved by the declarative frontend")
    shadowed = sorted(SCHEME_NAMES.intersection(aliases))
    if shadowed:
        raise HNDLError("E_NAME", "Operator aliases shadow initializer schemes reserved by the "
                                  f"declarative frontend: {', '.join(shadowed)}")
    dedented = textwrap.dedent(source)
    original_lines = source.split("\n")
    clean_lines = dedented.split("\n")
    offsets = [max(0, len(original) - len(clean)) for original, clean in zip(original_lines, clean_lines)]
    _screen(dedented, offsets)
    try:
        with warnings.catch_warnings():
            # Invalid escape sequences and the like are the literal's business.
            warnings.simplefilter("ignore")
            module = ast.parse(dedented, mode="exec", feature_version=(3, 11))
    except SyntaxError as exc:
        line = exc.lineno or 1
        offset = offsets[line - 1] if line <= len(offsets) else 0
        raise HNDLError("E_SYNTAX", exc.msg, line=line, column=(exc.offset or 1) + offset) from None
    except ValueError as exc:  # null bytes on Python 3.11
        raise HNDLError("E_SYNTAX", str(exc)) from None
    except (MemoryError, RecursionError):
        # How CPython's parser reports nesting the screen let through.
        raise HNDLError("E_RESOURCE", "Source is nested too deeply to parse") from None
    pending = [(module, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > MAX_AST_DEPTH:
            raise HNDLError("E_RESOURCE", f"Source nests expressions more than {MAX_AST_DEPTH} levels deep",
                            line=getattr(node, "lineno", None))
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return {"statements": Validator(aliases, offsets, clean_lines).statements(module.body)}
