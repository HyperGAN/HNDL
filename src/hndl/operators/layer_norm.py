from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    """A handwritten ``F.layer_norm`` equivalent of the built module."""
    normalized_shape, eps = module.normalized_shape, module.eps
    weight, bias = module.weight, module.bias

    def layer_norm(x):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    return layer_norm


@operator(
    "layer_norm",
    summary="Normalize the last axis of every position with a learned scale and bias.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    args={
        "eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False,
                   help="Added to the variance before the square root; must be positive."),
        "affine": Arg(bool, True, positional=False,
                      help="Learn a per-feature scale and bias of width D. When false the layer has no parameters."),
    },
    reference=_reference,
    examples=[
        Example("linear(64)\nlayer_norm()\nrelu()\nlinear()", ("B", 32), ("B", 10),
                "The usual placement: normalize the hidden width before the activation."),
        Example("linear(64)\nlayer_norm()\nlinear()", ("B", 12, 32), ("B", 12, 10),
                "On a [B, T, D] sequence each of the 12 positions is normalized independently."),
        Example("layer_norm(eps=0.001, affine=False)\nlinear()", ("B", 16), ("B", 4),
                "A parameter-free normalization of the incoming features."),
    ],
    category="normalization",
)
class LayerNorm(nn.LayerNorm):
    """Normalizes each row of the last axis ``D`` to zero mean and unit
    variance, then applies a learned per-feature affine:

    ```text
    mean = mean(x, dim=-1)
    var  = mean((x - mean)^2, dim=-1)            # biased: divided by D
    out  = (x - mean) / sqrt(var + eps) * weight + bias
    ```

    The statistics are taken over the last axis only, so a ``[B, D]`` input
    normalizes each example and a ``[B, T, D]`` sequence normalizes each
    position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
    relation but would be normalized over ``W`` alone, which is almost never
    what is wanted; reach for `group_norm` on images. The examples below
    therefore cover only ranks 2 and 3.

    Parameters are ``weight`` (initialized to ones) and ``bias``
    (initialized to zeros), both of shape ``[D]``, and exist only when
    ``affine`` is true. There are no running statistics, so train and eval
    behave identically. The computation runs in the plan's compute dtype.
    """

    def __init__(self, eps, affine, *, D):
        super().__init__(D, eps=eps, elementwise_affine=affine)
