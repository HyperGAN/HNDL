"""Ternary conditional selection operator."""

import torch
from torch import nn

from ..operator import Example, operator


def _relation(s):
    cond_dtype = s.dtype("cond")
    if cond_dtype is not None and cond_dtype != "bool":
        s.error("E_DTYPE", f"where condition tensor must be bool (got {cond_dtype})")

    s.share_batch("cond", "a", "b", "out")
    s.equal("cond", "a")
    s.equal("a", "b")
    s.equal("b", "out")


def _reference(module):
    def run(cond, a, b):
        return torch.where(cond, a, b)

    return run


@operator(
    "where",
    identity="where",
    summary="Ternary conditional selection: out = where(cond, a, b).",
    shape="cond[B, ...]:bool, a[B, ...], b[B, ...] -> out[B, ...]",
    relation=_relation,
    args={},
    examples=[
        Example(
            "b = linear(8)\nmask = cast(x, 'bool')\nout = where(mask, x, b)",
            ("B", 8),
            ("B", 8),
            "Selects elements from x where mask is True, and b where False.",
        ),
    ],
    reference=_reference,
    category="core",
)
class Where(nn.Module):
    """Ternary element-wise selection: out = torch.where(cond, a, b)."""

    def __init__(self):
        super().__init__()

    def forward(self, cond, a, b):
        return torch.where(cond, a, b)
