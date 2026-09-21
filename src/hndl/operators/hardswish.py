from torch import nn
from torch.nn import functional as F

from ..operator import Example, operator


def _reference(module):
    def hardswish(x):
        return F.hardswish(x)

    return hardswish


@operator(
    "hardswish",
    summary="Piecewise-linear approximation of swish, x * relu6(x + 3) / 6.",
    shape="x[B, ...] -> out[B, ...]",
    examples=[
        Example("conv(8, kernel_size=3, padding=1)\nhardswish()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "The activation used by mobile-scale image backbones."),
        Example("linear(64)\nhardswish()\nlinear()", ("B", 32), ("B", 10),
                "A cheap smooth-ish alternative to relu() in a classifier head."),
        Example("linear(16)\nhardswish()", ("B", 4, 8), ("B", 4, 16),
                "Elementwise on a [B, T, D] sequence."),
    ],
    category="activation",
    reference=_reference,
)
class Hardswish(nn.Hardswish):
    """Elementwise piecewise-linear approximation of `swish = x * sigmoid(x)`:

    ```
    out = 0                    if x <= -3
    out = x                    if x >= +3
    out = x * (x + 3) / 6      otherwise
    ```

    Equivalently `out = x * relu6(x + 3) / 6`. The hard sigmoid gate needs
    only clamping and a multiply, which makes this noticeably cheaper than
    `swish` on hardware without a fast `exp`, while tracking it closely. The
    function is continuous, its derivative is discontinuous at `x = -3` and
    `x = +3`, and negative inputs below -3 are zeroed exactly.

    The operation is computed out of place in the activation dtype
    (float32, float16, or bfloat16) and preserves any supported shape — rank 2
    `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`. It has no
    parameters and behaves identically in train and eval mode.
    """

    def __init__(self):
        super().__init__(inplace=False)
