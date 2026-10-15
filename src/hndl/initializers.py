"""Random initializer schemes for ``init={path: scheme}`` metadata.

Each function mirrors one ``torch.nn.init`` routine, keeping torch's keyword
names and defaults, and returns the canonical scheme record a plan stores.
The declarative frontend writes the same calls in configuration source, so
``ops.linear(init={"weight": xavier_uniform()})`` and
``linear(init={"weight": xavier_uniform()})`` resolve to identical plans.

These functions import no torch and touch no capture state; the backend
applies the selected scheme inside the construction RNG scope at build time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .settings import initializer


def xavier_uniform(*, gain: float = 1.0) -> Mapping[str, Any]:
    """``torch.nn.init.xavier_uniform_``; requires a parameter of rank >= 2."""
    return initializer("xavier_uniform", {"gain": gain})


def xavier_normal(*, gain: float = 1.0) -> Mapping[str, Any]:
    """``torch.nn.init.xavier_normal_``; requires a parameter of rank >= 2."""
    return initializer("xavier_normal", {"gain": gain})


def kaiming_uniform(
    *, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> Mapping[str, Any]:
    """``torch.nn.init.kaiming_uniform_``; requires a parameter of rank >= 2."""
    return initializer("kaiming_uniform", {"a": a, "mode": mode, "nonlinearity": nonlinearity})


def kaiming_normal(
    *, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> Mapping[str, Any]:
    """``torch.nn.init.kaiming_normal_``; requires a parameter of rank >= 2."""
    return initializer("kaiming_normal", {"a": a, "mode": mode, "nonlinearity": nonlinearity})


def truncated_normal(
    *, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Mapping[str, Any]:
    """``torch.nn.init.trunc_normal_``: normal draws resampled into ``[a, b]``."""
    return initializer("truncated_normal", {"mean": mean, "std": std, "a": a, "b": b})


def normal(*, mean: float = 0.0, std: float = 1.0) -> Mapping[str, Any]:
    """``torch.nn.init.normal_``."""
    return initializer("normal", {"mean": mean, "std": std})


def uniform(*, a: float = 0.0, b: float = 1.0) -> Mapping[str, Any]:
    """``torch.nn.init.uniform_``."""
    return initializer("uniform", {"a": a, "b": b})


def orthogonal(*, gain: float = 1.0) -> Mapping[str, Any]:
    """``torch.nn.init.orthogonal_``; requires a parameter of rank >= 2."""
    return initializer("orthogonal", {"gain": gain})


__all__ = ["kaiming_normal", "kaiming_uniform", "normal", "orthogonal", "truncated_normal",
           "uniform", "xavier_normal", "xavier_uniform"]
