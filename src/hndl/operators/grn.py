"""Global Response Normalization (GRN) for ConvNeXt-V2."""

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _reference(module):
    gamma, beta, eps = module.gamma, module.beta, module.eps

    def run(x):
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_calc = x.to(calc_dtype)
        gx = torch.sqrt(torch.sum(x_calc**2, dim=(2, 3), keepdim=True) + eps)
        nx = (gx / (gx.mean(dim=1, keepdim=True) + eps)).to(x.dtype)
        return gamma * (x * nx) + beta + x

    return run


@operator(
    "grn",
    summary="Global Response Normalization (GRN) for ConvNeXt-V2.",
    shape="x[B, C, H, W] -> out[B, C, H, W]",
    args={
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
            "conv(32, kernel_size=3, padding=1)\ngrn()\nconv(10, kernel_size=1)",
            ("B", 16, 8, 8),
            ("B", 10, 8, 8),
            "Applies global response normalization across feature channels.",
        ),
    ],
    reference=_reference,
    category="normalization",
)
class GRN(nn.Module):
    """Global Response Normalization (GRN) for ConvNeXt-V2.

    Computes spatial L2 norm per channel, normalizes across the channel axis,
    and recalibrates features via zero-initialized learned scale and bias:
    ``Gx = sqrt(sum(x^2, dim=(2, 3)) + eps)``
    ``Nx = Gx / (mean(Gx, dim=1) + eps)``
    ``out = gamma * (x * Nx) + beta + x``.
    """

    def __init__(self, eps=1e-6, *, C):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, int(C), 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, int(C), 1, 1))

    def forward(self, x):
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_calc = x.to(calc_dtype)
        gx = torch.sqrt(torch.sum(x_calc**2, dim=(2, 3), keepdim=True) + self.eps)
        nx = (gx / (gx.mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        return self.gamma * (x * nx) + self.beta + x

    def extra_repr(self):
        return f"eps={self.eps}"
