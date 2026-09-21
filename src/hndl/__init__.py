"""HNDL: human-readable network definitions with explicit shape constraints."""

from ._version import __version__
from .capture import ops, resolve_callable
from .config import resolve, resolve_file
from .errors import HNDLError
from .initializers import (kaiming_normal, kaiming_uniform, normal, orthogonal,
                           truncated_normal, uniform, xavier_normal, xavier_uniform)
from .operator import Arg, Example, Policy, operator
from .registry import Registry
from .types import ResolvedPlan

__all__ = ["__version__", "Arg", "Example", "HNDLError", "Policy", "Registry", "ResolvedPlan",
           "kaiming_normal", "kaiming_uniform", "normal", "operator", "ops", "orthogonal",
           "resolve", "resolve_callable", "resolve_file", "truncated_normal", "uniform",
           "xavier_normal", "xavier_uniform"]
