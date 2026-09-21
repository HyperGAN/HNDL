import torch
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, operator


def _validate(args):
    if args["min"] > args["max"]:
        raise HNDLError("E_ARGUMENT", f"clamp requires min <= max, got min={args['min']} and max={args['max']}")


def _reference(module):
    def clamp(x):
        return torch.clamp(x, min=module.min, max=module.max)
    return clamp


@operator(
    "clamp",
    summary="Saturate every element into the closed interval [min, max].",
    shape="x[B, ...] -> out[B, ...]",
    args={
        "min": Arg(float, help="Lower bound; elements below it become min."),
        "max": Arg(float, help="Upper bound; elements above it become max. Must be >= min."),
    },
    validate=_validate,
    reference=_reference,
    examples=[
        Example("linear(64)\nclamp(0.0, 6.0)\nlinear()", ("B", 128), ("B", 10),
                "A ReLU6-style activation on the hidden features."),
        Example("conv(8, kernel_size=3, padding=1)\nclamp(-1.0, 1.0)", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Saturating an image tensor into [-1, 1]."),
        Example("linear(8)\nclamp(0.0, 1.0)\nlinear()", ("B", 4, 6), ("B", 4, 3),
                "On a [B, T, D] sequence the bounds apply to every element."),
    ],
    category="activation",
)
class Clamp(nn.Module):
    """Elementwise ``out = min(max(x, min), max)``, computed as
    ``torch.clamp(x, min=min, max=max)`` out of place.

    The operation is shape preserving: every supported rank (``[B, F]``,
    ``[B, T, D]``, ``[B, C, H, W]``) passes through unchanged, and no axis has
    a special meaning. Both bounds are required floats and ``min <= max`` is
    checked when arguments are normalized (``E_ARGUMENT`` otherwise).

    The gradient is one strictly inside the interval and zero outside it, so a
    saturated element stops contributing to its input's gradient; elements
    exactly on a bound keep a gradient of one, matching ``torch.clamp``.
    Behavior is identical in train and eval mode and there are no parameters.
    Computation stays in the input dtype, so ``float16`` and ``bfloat16``
    activations are clamped without an upcast.
    """

    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

    def extra_repr(self):
        return f"min={self.min}, max={self.max}"
