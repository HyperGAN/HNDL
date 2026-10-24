from typing import Any

from ._version import __version__
from .capture import Symbol, ops, resolve_callable
from .config import resolve, resolve_file
from .errors import HNDLError
from .initializers import (kaiming_normal, kaiming_uniform, normal, orthogonal,
                           truncated_normal, uniform, xavier_normal, xavier_uniform)
from .operator import Arg, Example, NodeView, Policy, operator
from .profile import PlanProfile
from .registry import Registry
from .types import PlanDiff, ResolvedPlan

__all__ = [
    "Arg",
    "Example",
    "HNDLError",
    "NodeView",
    "PlanDiff",
    "PlanProfile",
    "Policy",
    "Registry",
    "ResolvedPlan",
    "Symbol",
    "__version__",
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "operator",
    "ops",
    "orthogonal",
    "resolve",
    "resolve_callable",
    "resolve_file",
    "torch",
    "truncated_normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
]


def __getattr__(name: str) -> Any:
    if name == "torch":
        import hndl.torch as torch_module

        return torch_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
