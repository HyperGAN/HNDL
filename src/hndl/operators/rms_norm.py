import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    """A handwritten ``F.rms_norm`` equivalent of the built module."""
    normalized_shape, eps = module.normalized_shape, module.eps
    weight = module.weight

    def rms_norm(x):
        return F.rms_norm(x, normalized_shape, weight, eps)

    return rms_norm


@operator(
    "rms_norm",
    summary="Scale the last axis by its root-mean-square, with a learned per-feature gain.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    args={
        "eps": Arg(float, 1e-6, min=0, exclusive_min=True, positional=False,
                   help="Added to the mean square before the reciprocal square root; must be positive."),
        "affine": Arg(bool, True, positional=False,
                      help="Learn a per-feature gain of width D. When false the layer has no parameters."),
    },
    reference=_reference,
    examples=[
        Example("linear(64)\nrms_norm()\nrelu()\nlinear()", ("B", 32), ("B", 10),
                "Pre-activation normalization without the mean subtraction of `layer_norm`."),
        Example("linear(64)\nrms_norm()\nlinear()", ("B", 12, 32), ("B", 12, 10),
                "On a [B, T, D] sequence each of the 12 positions is normalized independently."),
        Example("rms_norm(eps=0.0001, affine=False)\nlinear()", ("B", 16), ("B", 4),
                "A parameter-free variant, useful directly on the network input."),
    ],
    category="normalization",
)
class RMSNorm(nn.Module):
    """Root-mean-square normalization of the last axis ``D``: the cheaper
    half of `layer_norm` that rescales without re-centering.

    ```text
    out = x / sqrt(mean(x^2, dim=-1) + eps) * weight
    ```

    The statistics are taken over the last axis only, so a ``[B, D]`` input
    normalizes each example and a ``[B, T, D]`` sequence normalizes each
    position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
    relation but would be normalized over ``W`` alone, which is almost never
    what is wanted; reach for `group_norm` on images. The examples below
    therefore cover only ranks 2 and 3.

    The mean square and its reciprocal square root are accumulated in
    ``float32`` even when the plan's compute dtype is ``float16`` or
    ``bfloat16``, because squaring a half-precision activation overflows
    around 256; the normalized tensor is cast back to the input dtype before
    the gain is applied, so the layer's inputs, outputs and parameters stay
    in the plan dtype.

    The single parameter is ``weight`` of shape ``[D]``, initialized to ones
    and present only when ``affine`` is true. There are no running
    statistics, so train and eval behave identically.
    """

    def __init__(self, eps, affine, *, D):
        super().__init__()
        self.normalized_shape = (int(D),)
        self.eps = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(int(D)))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        statistic = x.float()
        scale = torch.rsqrt(statistic.square().mean(dim=-1, keepdim=True) + self.eps)
        out = (statistic * scale).to(x.dtype)
        return out if self.weight is None else out * self.weight

    def extra_repr(self):
        return f"{self.normalized_shape[0]}, eps={self.eps}, affine={self.weight is not None}"
