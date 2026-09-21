"""Random initializer schemes for ``init={path: scheme}`` metadata.

Each function mirrors one ``torch.nn.init`` routine, keeping torch's keyword
names and defaults, and returns the canonical scheme record a plan stores.
The declarative frontend writes the same calls in configuration source, so
``ops.linear(init={"weight": xavier_uniform()})`` and
``linear(init={"weight": xavier_uniform()})`` resolve to identical plans.

These functions import no torch and touch no capture state; the backend
applies the selected scheme inside the construction RNG scope at build time.
"""

from .settings import initializer


def xavier_uniform(*, gain=1.0):
    """``torch.nn.init.xavier_uniform_``; requires a parameter of rank >= 2."""
    return initializer("xavier_uniform", {"gain": gain})


def xavier_normal(*, gain=1.0):
    """``torch.nn.init.xavier_normal_``; requires a parameter of rank >= 2."""
    return initializer("xavier_normal", {"gain": gain})


def kaiming_uniform(*, a=0.0, mode="fan_in", nonlinearity="leaky_relu"):
    """``torch.nn.init.kaiming_uniform_``; requires a parameter of rank >= 2."""
    return initializer("kaiming_uniform", {"a": a, "mode": mode, "nonlinearity": nonlinearity})


def kaiming_normal(*, a=0.0, mode="fan_in", nonlinearity="leaky_relu"):
    """``torch.nn.init.kaiming_normal_``; requires a parameter of rank >= 2."""
    return initializer("kaiming_normal", {"a": a, "mode": mode, "nonlinearity": nonlinearity})


def truncated_normal(*, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """``torch.nn.init.trunc_normal_``: normal draws resampled into ``[a, b]``."""
    return initializer("truncated_normal", {"mean": mean, "std": std, "a": a, "b": b})


def normal(*, mean=0.0, std=1.0):
    """``torch.nn.init.normal_``."""
    return initializer("normal", {"mean": mean, "std": std})


def uniform(*, a=0.0, b=1.0):
    """``torch.nn.init.uniform_``."""
    return initializer("uniform", {"a": a, "b": b})


def orthogonal(*, gain=1.0):
    """``torch.nn.init.orthogonal_``; requires a parameter of rank >= 2."""
    return initializer("orthogonal", {"gain": gain})


__all__ = ["kaiming_normal", "kaiming_uniform", "normal", "orthogonal", "truncated_normal",
           "uniform", "xavier_normal", "xavier_uniform"]
