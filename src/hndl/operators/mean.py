from torch import nn

from ..operator import Arg, Example, SUPPORTED_RANKS, operator
from ._relations import reduce_axis


def _reference(module):
    def mean(x):
        return x.mean(dim=module.dim, keepdim=module.keepdim)
    return mean


@operator(
    "mean",
    summary="Average one non-batch axis away.",
    shape="x -> out",
    relation=reduce_axis,
    shape_text="out drops axis dim, or holds 1 there with keepdim; every other axis is unchanged",
    args={
        "dim": Arg(int, min=1, max=max(SUPPORTED_RANKS) - 1,
                   help="Axis to average over. The batch axis 0 cannot be reduced, so dim starts at 1."),
        "keepdim": Arg(bool, False, positional=False,
                       help="Keep the reduced axis with extent 1 instead of dropping it."),
    },
    reference=_reference,
    examples=[
        Example("linear(8)\nmean(1)", ("B", 4, 16), ("B", 8),
                "Mean-pools a [B, T, D] sequence over its positions into [B, D]."),
        Example("conv(8, kernel_size=3, padding=1)\nmean(2)\nmean(2)", ("B", 3, 8, 8), ("B", 8),
                "Two reductions average an image over H and then W: global average pooling."),
        Example("mean(2, keepdim=True)", ("B", 3, 8, 8), ("B", 3, 1, 8),
                "With keepdim the reduced axis stays with extent 1, so the rank is preserved."),
        Example("mean(1)\nlinear()", ("B", 6, 32), ("B", 10),
                "The features surviving the reduction are inferred backward from the output contract."),
    ],
    category="shape",
)
class Mean(nn.Module):
    """Arithmetic mean over exactly one non-batch axis:
    ``out = x.mean(dim=dim, keepdim=keepdim)``.

    With ``keepdim=False`` (the default) the output rank is one lower than the
    input rank and the axes after ``dim`` shift down by one. With
    ``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

    Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
    ``[B, T, D]`` (``mean(1)`` pools over positions, ``mean(2)`` over
    features), rank 4 is ``[B, C, H, W]`` (``mean(1)`` pools over channels,
    ``mean(2)`` over height, ``mean(3)`` over width), and a 1-D convolution
    layout ``[B, C, L]`` reduces its length with ``mean(2)``.

    Because HNDL tensors always keep a batch axis and at least one further
    axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
    ``E_CONSTRAINT``; use ``keepdim=True`` instead.

    The layer has no parameters and behaves identically in train and eval
    mode. The reduction runs in the incoming dtype — under ``float16`` a long
    axis accumulates rounding error, which ``sum`` shares.

    Shape inference is bidirectional: a known output fixes every non-reduced
    input axis, but the extent of the reduced axis itself carries no
    information backward and must come from the input side.
    """

    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self):
        return f"dim={self.dim}, keepdim={self.keepdim}"
