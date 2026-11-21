"""Collect the Python definitions needed by a standalone PyTorch export."""

import ast
import builtins
import copy
from functools import lru_cache
from importlib.util import resolve_name
import inspect
import symtable
import sys
import textwrap


def _global_names(source):
    """Names read from module scope, including in nested functions and classes."""
    root = symtable.symtable("from __future__ import annotations\n" + source, "<standalone>", "exec")
    tables = [root]
    names = set()
    while tables:
        table = tables.pop()
        names.update(s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced())
        tables.extend(table.get_children())
    return names


def _declared_operator(decorator):
    return isinstance(decorator, ast.Call) and (
        isinstance(decorator.func, ast.Name) and decorator.func.id == "operator"
        or isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "operator"
    )


def _definition(node):
    result = copy.deepcopy(node)
    if isinstance(result, ast.ClassDef):
        result.decorator_list = [d for d in result.decorator_list if not _declared_operator(d)]
    return result


@lru_cache(maxsize=128)
def _definitions(module):
    return top_level_bindings(ast.parse(inspect.getsource(module)))


def top_level_bindings(tree):
    """Index top-level bindings, preserving the originating import for each name."""
    bindings = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            bindings[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                single = copy.copy(node)
                single.names = [alias]
                bindings[bound] = single
    return bindings


class _ScopeGlobals(ast.NodeTransformer):
    def visit_Global(self, statement):
        return ast.Nonlocal(names=statement.names)


class SourceBundle:
    """Emit helpers once per defining module, keeping private names lexically scoped."""

    def __init__(self):
        self.modules = {}
        self.dependencies = {}
        self.aliases = {}

    def _alias(self, module):
        return self.aliases.setdefault(module, f"_source_{len(self.aliases) + 1}")

    def include(self, obj):
        module = inspect.getmodule(obj)
        if module is None:
            raise ValueError(f"source module unavailable for {obj!r}")
        # User registrations may contain classes declared inside a function.
        local = None
        if "<locals>" in obj.__qualname__:
            local = ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]
        self._require(module, obj.__name__, local)
        return f"{self._alias(module)}.{obj.__name__}"

    def _require(self, module, name, local=None):
        items = self.modules.setdefault(module, {})
        if name in items:
            if local is not None and ast.dump(local) != items[name][2]:
                raise ValueError(f"conflicting local definitions for {name!r}")
            return
        node = local or _definitions(module).get(name)
        if node is None:
            if hasattr(builtins, name) and name not in vars(module):
                return
            raise ValueError(f"cannot reproduce global {name!r} in {module.__name__}")
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            alias = node.names[0]
            path = alias.name if isinstance(node, ast.Import) else resolve_name(
                "." * node.level + (node.module or ""), module.__package__)
            package = path.split(".")[0]
            if package == "hndl":
                if isinstance(node, ast.Import) or not node.module:
                    raise ValueError(f"cannot embed module import {ast.unparse(node)}")
                other = sys.modules[path]
                self.dependencies.setdefault(module, set()).add(other)
                self._require(other, alias.name)
                source = f"{name} = {self._alias(other)}.{alias.name}"
            elif package == "torch" or package in sys.stdlib_module_names:
                source = ast.unparse(node)
            else:
                raise ValueError(f"external dependency {path!r} is required")
            items[name] = (node.lineno, source, None)
            return
        definition = _definition(node)
        source = ast.unparse(definition)
        items[name] = (node.lineno, source, ast.dump(local) if local is not None else None)
        for nested in ast.walk(definition):
            if isinstance(nested, ast.ImportFrom):
                packages = [(nested.module or "").split(".")[0]] if not nested.level else [""]
            elif isinstance(nested, ast.Import):
                packages = [alias.name.split(".")[0] for alias in nested.names]
            else:
                continue
            if any(package != "torch" and package not in sys.stdlib_module_names for package in packages):
                raise ValueError(f"cannot embed local import {ast.unparse(nested)}")
        for dep in sorted(_global_names(source) - {name}):
            self._require(module, dep)
        # The enclosing factory owns this module's state. A helper updating
        # a module-level cache must share that binding with its sibling helpers.
        for statement in ast.walk(definition):
            if isinstance(statement, ast.Global):
                for var in statement.names:
                    self._require(module, var)
        source = ast.unparse(_ScopeGlobals().visit(definition))
        items[name] = (node.lineno, source, items[name][2])

    def render(self):
        from graphlib import TopologicalSorter

        order = TopologicalSorter({m: self.dependencies.get(m, ()) for m in self.modules})
        lines = []
        for module in order.static_order():
            alias = self._alias(module)
            items = self.modules[module]
            lines.append(f"def {alias}_definitions():")
            # Nested local classes may come from position 1 in inspect.getsource.
            # Their bases and imports must already exist when the class is defined.
            imports = [n for n in items if isinstance(_definitions(module).get(n), (ast.ImportFrom, ast.Import))]
            others = sorted((n for n in items if n not in imports), key=lambda n: items[n][0])
            for name in [*imports, *others]:
                lines.append(textwrap.indent(items[name][1], "    "))
            members = ", ".join(f"{n}={n}" for n in items)
            lines += [f"    return SimpleNamespace({members})", "", f"{alias} = {alias}_definitions()", ""]
        return lines
