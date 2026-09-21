from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    approximate = module.approximate

    def gelu(x):
        return F.gelu(x, approximate=approximate)

    return gelu


@operator(
    "gelu",
    summary="Gaussian error linear unit, x * Phi(x).",
    shape="x[B, ...] -> out[B, ...]",
    args={
        "approximate": Arg(str, "none", choices=("none", "tanh"),
                           help='Formula to use: "none" for the exact erf form, "tanh" for the tanh approximation.'),
    },
    examples=[
        Example("linear(64)\ngelu()\nlinear()", ("B", 128), ("B", 10),
                "The usual transformer feed-forward activation on [B, F] features."),
        Example('linear(32)\ngelu(approximate="tanh")\nlinear()', ("B", 6, 16), ("B", 6, 8),
                "On a [B, T, D] sequence the activation applies elementwise at every position."),
        Example("conv(8, kernel_size=3, padding=1)\ngelu()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Elementwise over every channel and spatial position of a [B, C, H, W] tensor."),
    ],
    reference=_reference,
    category="activation",
)
class GELU(nn.GELU):
    """Elementwise ``out = x * Phi(x)``, where ``Phi`` is the standard normal
    cumulative distribution function.

    With ``approximate="none"`` (the default) the exact form is computed:

    ```text
    out = 0.5 * x * (1 + erf(x / sqrt(2)))
    ```

    With ``approximate="tanh"`` the cheaper tanh approximation is used, which
    is what the original BERT and GPT-2 implementations shipped:

    ```text
    out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ```

    The two forms agree to roughly 1e-3 in absolute value but are not
    interchangeable when loading weights trained against a specific one, so
    the choice is recorded in the plan.

    The operator is elementwise and shape preserving on ``[B, F]``,
    ``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
    identically in train and eval mode, and is computed out of place in the
    plan's compute dtype.
    """

    def __init__(self, approximate):
        super().__init__(approximate=approximate)
