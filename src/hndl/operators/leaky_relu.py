from torch import nn

from ..operator import Arg, Example, operator


@operator(
    "leaky_relu",
    summary="ReLU with a small slope for negative inputs.",
    shape="x[B, ...] -> out[B, ...]",
    args={"negative_slope": Arg(float, 0.01, help="Multiplier applied to negative inputs.")},
    examples=[Example("conv(8, kernel_size=3, padding=1)\nleaky_relu(0.2)", ("B", 3, 8, 8), ("B", 8, 8, 8))],
    category="activation",
)
class LeakyReLU(nn.LeakyReLU):
    """``out = x if x >= 0 else negative_slope * x``, computed out of place."""

    def __init__(self, negative_slope):
        super().__init__(negative_slope, inplace=False)
