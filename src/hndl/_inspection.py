"""Load built-in contracts for static CLI inspection without a tensor runtime.

Only the installed operator catalog is read here. The same decorator, argument
normalization, and relation functions are used by the ordinary runtime registry.
"""

import ast
from functools import lru_cache
import importlib
from pathlib import Path

from ._source_bundle import top_level_bindings
from .registry import Registry


class _DeferredModule:
    """Keep the constructor signature for resolution, load the real type on construction."""

    def __new__(cls, *args, **kwargs):
        module = importlib.import_module(cls.__module__)
        return getattr(module, cls.__name__)(*args, **kwargs)

    def forward(self):
        raise AssertionError("Construction must return the runtime operator")


def _declaration(tree):
    """Select the decorated class signature, relations and their dependencies."""
    bindings = top_level_bindings(tree)
    roots = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            # Some declarations specialize their output ports after decoration.
            roots.append(node)
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = [d for d in node.decorator_list if isinstance(d, ast.Call)
                      and isinstance(d.func, ast.Name) and d.func.id == "operator"]
        if not decorators:
            continue
        for decorator in decorators:
            decorator.keywords = [kw for kw in decorator.keywords if kw.arg != "reference"]
        node.decorator_list = decorators
        node.bases = [ast.Name(id="_DeferredModule", ctx=ast.Load())]
        constructor = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "__init__"]
        for method in constructor:
            method.body = [ast.Pass()]
        node.body = constructor or [ast.Pass()]
        roots.append(node)
        classes.append(node.name)

    selected = set()
    while roots:
        node = roots.pop()
        if node in selected:
            continue
        selected.add(node)
        roots.extend(bindings[dep.id] for dep in ast.walk(node)
                     if isinstance(dep, ast.Name) and isinstance(dep.ctx, ast.Load) and dep.id in bindings)
    tree.body = sorted(selected, key=lambda n: n.lineno)
    return ast.fix_missing_locations(tree), classes


_CATALOG = Path(__file__).with_name("operators")


@lru_cache(maxsize=None)
def _module_schemas(filename):
    path = _CATALOG / (filename + ".py")
    tree, classes = _declaration(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    scope = {"__name__": f"hndl.operators.{filename}", "__package__": "hndl.operators",
             "_DeferredModule": _DeferredModule}
    exec(compile(tree, str(path), "exec"), scope)
    return {scope[name].__hndl_operator__.alias: scope[name].__hndl_operator__ for name in classes}


@lru_cache(maxsize=1)
def _catalog_aliases():
    return tuple(sorted(p.stem for p in _CATALOG.glob("*.py") if not p.name.startswith("_")))


class InspectionRegistry(Registry):
    """CLI registry with lazy schema loading for the package's one-file-per-op catalog."""

    @property
    def aliases(self):
        return _catalog_aliases()

    def get(self, alias):
        if alias not in self._aliases and alias in self.aliases:
            self._add(_module_schemas(alias)[alias])
        return super().get(alias)
