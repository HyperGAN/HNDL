import torch
from torch import nn

from ..operator import Example, operator


@operator(
    "sub",
    summary="Elementwise difference of two tensors with identical shapes.",
    shape="a[B, ...], b[B, ...] -> out[B, ...]",
    reference=lambda module: torch.sub,
    examples=[
        Example('saved = x\nlinear(8, name="branch")\nrelu()\nh = linear(4)\nsub(h, saved)',
                ("B", 4), ("B", 4), "A residual that subtracts the shortcut; both inputs are explicit."),
        Example("a, b = split(4, dim=2)\nsub(a, b)", ("B", 6, 8), ("B", 6, 4),
                "Sequences work too: the two halves of the feature axis are differenced."),
        Example("saved = x\nconv(3, kernel_size=3, padding=1)\nh = tanh()\nsub(h, saved)",
                ("B", 3, 8, 8), ("B", 3, 8, 8), "Images: the operand shapes must match exactly."),
    ],
    category="arithmetic",
)
class Sub(nn.Module):
    """``out = a - b`` with no broadcasting; the two operands must agree on
    rank, every axis extent, and dtype, and the difference is computed in the
    plan compute dtype. Order matters, so both tensor inputs must be supplied
    explicitly: ``sub(h, saved)`` is ``h - saved``.
    """

    def forward(self, a, b):
        return a - b
