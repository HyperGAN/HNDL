from torch import nn
from torch.nn import functional as F

from ..operator import Example, operator


def _reference(module):
    return F.silu


@operator(
    "silu",
    summary="Sigmoid linear unit (swish), x * sigmoid(x).",
    shape="x[B, ...] -> out[B, ...]",
    examples=[
        Example("linear(64)\nsilu()\nlinear()", ("B", 128), ("B", 10),
                "A smooth alternative to relu() on [B, F] features."),
        Example("linear(32)\nsilu()\nlinear()", ("B", 6, 16), ("B", 6, 8),
                "On a [B, T, D] sequence the activation applies elementwise at every position."),
        Example("conv(8, kernel_size=3, padding=1)\ngroup_norm(4)\nsilu()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "The norm-then-activation pairing used by diffusion U-Nets on [B, C, H, W] tensors."),
    ],
    reference=_reference,
    category="activation",
)
class SiLU(nn.SiLU):
    """Elementwise ``out = x * sigmoid(x)``, also known as swish.

    Unlike `relu` the function is smooth everywhere and keeps a small negative
    response, with a minimum of about ``-0.278`` near ``x = -1.278``; unlike
    `sigmoid` it is unbounded above, so it does not saturate for large
    positive inputs.

    The operator is elementwise and shape preserving on ``[B, F]``,
    ``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
    identically in train and eval mode, and is computed out of place in the
    plan's compute dtype.
    """

    def __init__(self):
        super().__init__(inplace=False)
