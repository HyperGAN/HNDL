import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, operator


def _validate(args):
    if args.get("zero_centered") and not args.get("affine"):
        raise HNDLError("E_ARGUMENT", "zero_centered=True requires affine=True")


def _reference(module):
    """Explicit reciprocal-square-root mean square formula without fused kernel."""
    eps = module.eps
    weight = module.weight
    zero_centered = getattr(module, "zero_centered", False)

    def rms_norm(x):
        statistic = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(statistic.square().mean(dim=-1, keepdim=True) + eps)
        out = (statistic * scale).to(x.dtype)
        if weight is None:
            return out
        if zero_centered:
            return out * (1.0 + weight)
        return out * weight

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
        "zero_centered": Arg(bool, False, positional=False, since="0.9.0",
                             help="Initialize weight to 0.0 and evaluate as (1 + weight) * rms_norm(x)."),
    },
    validate=_validate,
    reference=_reference,
    examples=[
        Example("linear(64)\nrms_norm()\nrelu()\nlinear()", ("B", 32), ("B", 10),
                "Pre-activation normalization without the mean subtraction of `layer_norm`."),
        Example("linear(64)\nrms_norm()\nlinear()", ("B", 12, 32), ("B", 12, 10),
                "On a [B, T, D] sequence each of the 12 positions is normalized independently."),
        Example("rms_norm(eps=0.0001, affine=False)\nlinear()", ("B", 16), ("B", 4),
                "A parameter-free variant, useful directly on the network input."),
        Example("linear(64)\nrms_norm(zero_centered=True)\nlinear()", ("B", 32), ("B", 10),
                "Gemma-style zero-centered RMSNorm with weight initialized to 0.0."),
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
    (or zeros when ``zero_centered=True``) and present only when ``affine`` is
    true. There are no running statistics, so train and eval behave identically.
    """

    def __init__(self, eps, affine, *, D, zero_centered=False):
        super().__init__()
        self.normalized_shape = (int(D),)
        self.eps = float(eps)
        self.zero_centered = bool(zero_centered)
        if self.zero_centered and not affine:
            raise HNDLError("E_ARGUMENT", "zero_centered=True requires affine=True")
        if affine:
            if self.zero_centered:
                self.weight = nn.Parameter(torch.zeros(int(D)))
            else:
                self.weight = nn.Parameter(torch.ones(int(D)))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        if self.zero_centered:
            normed = F.rms_norm(x, self.normalized_shape, None, self.eps)
            return normed * (1.0 + self.weight)
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self):
        zc = f", zero_centered={self.zero_centered}" if self.zero_centered else ""
        return f"{self.normalized_shape[0]}, eps={self.eps}, affine={self.weight is not None}{zc}"
