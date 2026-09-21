"""Built-in operators: one module per operation, discovered by name.

Every module in this package defines one or more classes decorated with
``@operator``. ``discover()`` imports them in sorted order and returns their
declarations; ``Registry.builtins()`` binds those into a registry.
"""

import importlib
import pkgutil

from ..errors import HNDLError

_CACHE = None


def discover():
    global _CACHE
    if _CACHE is None:
        found = []
        for info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
            if info.name.startswith("_"):
                continue
            module = importlib.import_module(f"{__name__}.{info.name}")
            specs = [value.__hndl_operator__ for value in vars(module).values()
                     if isinstance(value, type) and "__hndl_operator__" in value.__dict__
                     and value.__module__ == module.__name__]
            if not specs:
                raise HNDLError("E_REGISTRY", f"Operator module {module.__name__} declares no @operator class")
            found.extend(specs)
        _CACHE = tuple(found)
    return _CACHE
