"""Parameter-free L2 normalization along a specified axis."""

import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    dim = module.dim
    eps = module.eps

    def run(x):
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(eps)
        return x / norm

    return run


@operator(
    "l2_norm",
    summary="Parameter-free L2 normalization along a specified axis.",
    shape="x[B, ...] -> out[B, ...]",
    args={
        "dim": Arg(int, -1, positional=False, help="Axis along which L2 normalization is computed."),
        "eps": Arg(
            float,
            1e-6,
            min=0,
            exclusive_min=True,
            positional=False,
            help="Small positive constant to avoid division by zero.",
        ),
    },
    examples=[
        Example(
            "linear(64)\nl2_norm()\nlinear()",
            ("B", 32),
            ("B", 10),
            "Normalizes hidden vectors along the feature dimension.",
        ),
    ],
    reference=_reference,
    category="normalization",
)
class L2Norm(nn.Module):
    """Normalize vectors along a specified dimension to unit L2 norm.

    Applies parameter-free L2 normalization:
    ``out = x / max(norm(x, 2, dim), eps)``.
    """

    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)
