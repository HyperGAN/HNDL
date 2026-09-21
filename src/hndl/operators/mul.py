import torch
from torch import nn

from ..operator import Example, operator


@operator(
    "mul",
    summary="Elementwise product of two tensors with identical shapes.",
    shape="a[B, ...], b[B, ...] -> out[B, ...]",
    reference=lambda module: torch.mul,
    examples=[
        Example("h = linear(4)\ng = linear(x, 4)\ng = tanh(g)\nmul(h, g)", ("B", 8), ("B", 4),
                "A gate: a second branch scales the first elementwise."),
        Example("a, b = split(4, dim=2)\nmul(a, b)", ("B", 6, 8), ("B", 6, 4),
                "Sequences work too: the two halves of the feature axis are multiplied."),
        Example("mask = x\nconv(3, kernel_size=3, padding=1)\nh = tanh()\nmul(h, mask)",
                ("B", 3, 8, 8), ("B", 3, 8, 8), "Images: the operand shapes must match exactly."),
    ],
    category="arithmetic",
)
class Mul(nn.Module):
    """``out = a * b`` (the Hadamard product) with no broadcasting; the two
    operands must agree on rank, every axis extent, and dtype, and the product
    is computed in the plan compute dtype. Both tensor inputs must be supplied
    explicitly. Use ``scale`` instead to multiply by a constant.
    """

    def forward(self, a, b):
        return a * b
