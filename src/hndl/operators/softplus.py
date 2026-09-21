from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    beta, threshold = module.beta, module.threshold

    def softplus(x):
        return F.softplus(x, beta=beta, threshold=threshold)

    return softplus


@operator(
    "softplus",
    summary="Smooth positive activation, log(1 + exp(beta*x)) / beta.",
    shape="x[B, ...] -> out[B, ...]",
    args={
        "beta": Arg(float, 1.0, min=0, exclusive_min=True,
                    help="Sharpness of the bend at zero; larger values approach ReLU."),
        "threshold": Arg(float, 20.0, min=0, exclusive_min=True, positional=False,
                         help="Above beta*x = threshold the function is evaluated as the identity for stability."),
    },
    examples=[
        Example("linear(64)\nsoftplus()\nlinear()", ("B", 32), ("B", 10),
                "A strictly positive hidden activation."),
        Example("conv(8, kernel_size=3, padding=1)\nsoftplus(2.0)", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "A sharper bend, closer to ReLU, on a [B, C, H, W] image."),
        Example("linear(16)\nsoftplus(1.0, threshold=10.0)", ("B", 4, 8), ("B", 4, 16),
                "Elementwise on a [B, T, D] sequence."),
    ],
    category="activation",
    reference=_reference,
)
class Softplus(nn.Softplus):
    """Elementwise smooth approximation of ReLU:

    ```
    out = log(1 + exp(beta * x)) / beta
    ```

    The output is strictly positive and the function is differentiable
    everywhere; its derivative is the logistic sigmoid `sigmoid(beta * x)`.
    Larger `beta` sharpens the bend at the origin and the limit is ReLU.

    For numerical stability the linear branch `out = x` is used wherever
    `beta * x > threshold`, which is exact in floating point well before the
    default threshold of 20. The rule applies unchanged in float16 and
    bfloat16, where the identity branch avoids overflowing `exp`.

    The operator is elementwise, so it preserves any supported shape — rank 2
    `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — and it has no
    parameters and no train/eval difference.
    """

    def __init__(self, beta, threshold):
        super().__init__(beta=beta, threshold=threshold)
