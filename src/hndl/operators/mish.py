from torch import nn
from torch.nn import functional as F

from ..operator import Example, operator


def _reference(module):
    def mish(x):
        return F.mish(x)

    return mish


@operator(
    "mish",
    summary="Self-gated smooth activation, x * tanh(softplus(x)).",
    shape="x[B, ...] -> out[B, ...]",
    examples=[
        Example("linear(64)\nmish()\nlinear()", ("B", 32), ("B", 10),
                "A smooth drop-in replacement for relu()."),
        Example("conv(8, kernel_size=3, padding=1)\nmish()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Elementwise on a [B, C, H, W] image."),
        Example("linear(16)\nmish()", ("B", 4, 8), ("B", 4, 16),
                "Elementwise on a [B, T, D] sequence."),
    ],
    category="activation",
    reference=_reference,
)
class Mish(nn.Mish):
    """Elementwise self-gated activation:

    ```
    out = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    ```

    The gate `tanh(softplus(x))` rises smoothly from 0 to 1, so `mish`
    approaches the identity for large positive inputs and decays towards 0 for
    large negative ones, with a small negative dip near `x = -1`. It is smooth
    everywhere (unlike ReLU) and unbounded above, which keeps gradients alive
    on the negative side.

    `softplus` inside the gate uses PyTorch's stable formulation, so the
    activation is safe in float16 and bfloat16 as well as float32; it is
    computed in the activation dtype without upcasting. It is elementwise, so
    it preserves any supported shape — rank 2 `[B, F]`, rank 3 `[B, T, D]`, or
    rank 4 `[B, C, H, W]` — has no parameters, and behaves identically in
    train and eval mode.
    """

    def __init__(self):
        super().__init__(inplace=False)
