from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    alpha = module.alpha

    def elu(x):
        return F.elu(x, alpha=alpha)

    return elu


@operator(
    "elu",
    summary="Exponential linear unit: identity above zero, saturating below.",
    shape="x[B, ...] -> out[B, ...]",
    args={
        "alpha": Arg(float, 1.0, help="Negative saturation value; the activation approaches -alpha as x decreases."),
    },
    examples=[
        Example("linear(64)\nelu()\nlinear()", ("B", 32), ("B", 10),
                "A hidden activation with nonzero gradient for negative inputs."),
        Example("conv(8, kernel_size=3, padding=1)\nelu(0.5)", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Half the negative saturation on a [B, C, H, W] image."),
        Example("linear(16)\nelu()", ("B", 4, 8), ("B", 4, 16),
                "Elementwise on a [B, T, D] sequence."),
    ],
    category="activation",
    reference=_reference,
)
class ELU(nn.ELU):
    """Elementwise exponential linear unit:

    ```
    out = x                          if x > 0
    out = alpha * (exp(x) - 1)       if x <= 0
    ```

    The function is continuous at the origin with value 0, and for negative
    inputs it saturates smoothly at `-alpha` instead of clamping to zero, so
    unlike ReLU it keeps a nonzero gradient there. `alpha = 1` gives the
    standard form with a continuous derivative at 0.

    The operation is computed out of place in the activation dtype
    (float32, float16, or bfloat16), preserves any supported shape — rank 2
    `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — has no
    parameters, and behaves identically in train and eval mode.
    """

    def __init__(self, alpha):
        super().__init__(alpha=alpha, inplace=False)
