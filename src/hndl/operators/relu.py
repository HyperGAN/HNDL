from torch import nn

from ..operator import Example, operator


@operator(
    "relu",
    summary="Rectified linear unit, max(x, 0).",
    shape="x[B, ...] -> out[B, ...]",
    examples=[Example("linear(64)\nrelu()\nlinear()", ("B", 128), ("B", 10))],
    category="activation",
)
class ReLU(nn.ReLU):
    """Out-of-place ReLU; shared branches are never mutated."""

    def __init__(self):
        super().__init__(inplace=False)
