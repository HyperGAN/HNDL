"""HNDL's torch-free authoring, registry, and resolution API."""

from ._version import __version__
from .capture import ops, resolve_callable
from .config import resolve, resolve_file
from .errors import HNDLError
from .registry import Registry, preserves_shape
from .schema import Argument, Dim, ShapeRule
from .types import ResolvedPlan

__all__ = ["__version__", "HNDLError", "Registry", "ResolvedPlan", "Argument", "Dim", "ShapeRule", "ops", "preserves_shape",
           "resolve", "resolve_callable", "resolve_file"]
