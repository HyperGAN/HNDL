from torch import nn

from ..operator import Example, operator


@operator(
    "tanh",
    summary="Hyperbolic tangent, squashing values into (-1, 1).",
    shape="x[B, ...] -> out[B, ...]",
    examples=[Example("conv(3, kernel_size=3, padding=1)\ntanh()", ("B", 8, 16, 16), ("B", 3, 16, 16),
                      "A common final activation for image generators.")],
    category="activation",
)
class Tanh(nn.Tanh):
    """Elementwise ``tanh(x)``."""
