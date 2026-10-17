"""Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers."""

import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    eps = module.eps
    linear = module.modulation

    def run(x, cond):
        params = F.linear(F.silu(cond), linear.weight, linear.bias)
        shift, scale, gate = params.chunk(3, dim=-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + eps)
        out = (1.0 + scale.unsqueeze(1)) * normed + shift.unsqueeze(1)
        gate_out = gate.unsqueeze(1).expand_as(x)
        return out, gate_out

    return run


@operator(
    "adaln_zero",
    summary="Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.",
    shape="x[B, T, D], cond[B, D_cond] -> out[B, T, D], gate[B, T, D]",
    args={
        "eps": Arg(
            float,
            1e-6,
            min=0,
            exclusive_min=True,
            positional=False,
            help="Epsilon added to the variance in LayerNorm.",
        ),
    },
    examples=[
        Example(
            "c = mean(1)\ncond = linear(c, 32)\nnormed, gate = adaln_zero(x, cond)\nmul(normed, gate)",
            ("B", 16, 64),
            ("B", 16, 64),
            "Modulates sequence features with conditioning vector and produces residual gate.",
        ),
    ],
    reference=_reference,
    category="normalization",
)
class AdaLNZero(nn.Module):
    """Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.

    Maps a conditioning vector ``cond[B, D_cond]`` through a zero-initialized
    linear layer to produce per-token shift, scale, and dimension-wise gate
    parameters:
    ``shift, scale, gate = Linear(SiLU(cond)).chunk(3, dim=-1)``
    ``normed = LayerNorm(x, affine=False)``
    ``out = (1 + scale) * normed + shift``
    ``gate_out = gate.expand_as(x)``.
    """

    def __init__(self, eps=1e-6, *, D, D_cond):
        super().__init__()
        self.eps = float(eps)
        self.D = int(D)
        self.D_cond = int(D_cond)
        self.modulation = nn.Linear(self.D_cond, 3 * self.D, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, cond):
        params = self.modulation(F.silu(cond))
        shift, scale, gate = params.chunk(3, dim=-1)
        normed = F.layer_norm(x, (self.D,), eps=self.eps)
        out = (1.0 + scale.unsqueeze(1)) * normed + shift.unsqueeze(1)
        gate_out = gate.unsqueeze(1).expand_as(x)
        return out, gate_out
