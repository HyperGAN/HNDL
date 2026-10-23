"""Stochastic depth (drop_path) operator."""

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        rank = 3
    else:
        rank = len(shape)
    s.rank("x", rank)
    s.rank("out", rank)
    s.equal("x", "out")


def _reference(module):
    drop_prob = module.drop_prob
    scale_by_keep = module.scale_by_keep

    def run(x):
        if not module.training or drop_prob == 0.0:
            return x
        if drop_prob == 1.0:
            return torch.zeros_like(x)
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = random_tensor < keep_prob
        if scale_by_keep:
            mask = mask.to(x.dtype) / keep_prob
        else:
            mask = mask.to(x.dtype)
        return x * mask

    return run


@operator(
    "drop_path",
    summary="Stochastic depth (DropPath) residual regularization.",
    shape="x[B, ...] -> out[B, ...]",
    relation=_relation,
    args={
        "drop_prob": Arg(float, 0.0, min=0.0, max=1.0, help="Probability of dropping residual path."),
        "scale_by_keep": Arg(bool, True, positional=False, help="Scale retained samples by 1 / (1 - drop_prob)."),
    },
    examples=[
        Example(
            "drop_path(0.0)",
            ("B", 8, 32),
            ("B", 8, 32),
            "Stochastic depth dropping residual connections (drop_prob=0.0 in examples for determinism).",
        ),
    ],
    reference=_reference,
    category="regularization",
)
class DropPath(nn.Module):
    """Stochastic depth layer (DropPath) per sample across batches."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        if self.drop_prob == 1.0:
            return torch.zeros_like(x)
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (random_tensor < keep_prob).to(x.dtype)
        if self.scale_by_keep:
            mask = mask / keep_prob
        return x * mask

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}, scale_by_keep={self.scale_by_keep}"
