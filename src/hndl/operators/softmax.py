import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        return
    dim = s.args["dim"]
    rank = len(shape)
    axis = dim + rank if dim < 0 else dim
    if not 0 <= axis < rank:
        s.error("E_ARGUMENT", f"dim={dim} is out of range for a rank-{rank} tensor")
    if axis == 0:
        s.error("E_ARGUMENT", f"dim={dim} selects the batch axis; softmax must normalize a non-batch axis")


def _reference(module):
    dim = module.dim

    def softmax(x):
        return F.softmax(x, dim=dim)

    return softmax


@operator(
    "softmax",
    summary="Normalize one non-batch axis into a probability distribution.",
    shape="x[B, ...] -> out[B, ...]",
    relation=_relation,
    shape_text="out has the shape of x; dim must resolve to a non-batch axis",
    args={
        "dim": Arg(int, -1, help="Axis to normalize. Negative values count from the end; the resolved "
                                 "axis must not be the batch axis 0."),
    },
    examples=[
        Example("linear(10)\nsoftmax()", ("B", 32), ("B", 10),
                "Class probabilities over the feature axis of a rank-2 tensor."),
        Example("linear(16)\nsoftmax(-1)", ("B", 4, 8), ("B", 4, 16),
                "A [B, T, D] sequence normalized over its feature axis D."),
        Example("conv(4, kernel_size=1)\nsoftmax(1)", ("B", 3, 4, 4), ("B", 4, 4, 4),
                "Per-pixel distribution across the channel axis of a [B, C, H, W] image."),
    ],
    category="activation",
    reference=_reference,
)
class Softmax(nn.Module):
    """Softmax along one axis:

    ```
    out[..., i, ...] = exp(x[..., i, ...] - m) / sum_j exp(x[..., j, ...] - m)
    ```

    where `m` is the maximum over the axis selected by `dim` and the sum runs
    over that same axis (subtracting the maximum is what `torch.softmax` does
    internally for numerical stability). Every slice along the axis sums to 1
    and every element lies in (0, 1). The shape is preserved.

    `dim` indexes the full tensor, batch axis included, following the usual
    PyTorch convention: `-1` is the last axis, `1` is the channel axis `C` of a
    `[B, C, H, W]` image or the position axis `T` of a `[B, T, D]` sequence,
    and `2` is `H`. Normalizing across the batch axis would make examples in a
    batch depend on each other, so a `dim` that resolves to axis 0, or that
    lies outside the rank of the incoming tensor, is rejected with
    `E_ARGUMENT` while the plan resolves.

    The computation runs in the activation dtype, including float16 and
    bfloat16. Behavior is identical in train and eval mode, and the operator
    has no parameters.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
