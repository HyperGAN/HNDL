"""HNDL: human-readable network definitions with explicit shape constraints."""

from ._version import __version__
from .capture import ops, resolve_callable
from .config import resolve, resolve_file
from .errors import HNDLError
from .operator import Arg, Example, Policy, operator
from .registry import Registry
from .types import ResolvedPlan

__all__ = ["__version__", "Arg", "Example", "HNDLError", "Policy", "Registry", "ResolvedPlan", "operator", "ops",
           "resolve", "resolve_callable", "resolve_file"]
