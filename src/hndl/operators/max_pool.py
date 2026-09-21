import torch.nn.functional as F
from torch import nn

from ..operator import Arg, Example, PAIR, operator
from ._pooling import normalize_stride, pooling, validate_padding


def _reference(module):
    def forward(x):
        return F.max_pool2d(x, module.kernel_size, stride=module.stride, padding=module.padding)
    return forward


@operator(
    "max_pool",
    identity="max_pool2d",
    summary="Two-dimensional max pooling over [B, C, H, W] images.",
    shape="x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]",
    relation=pooling(),
    shape_text="H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1, same for W; stride=0 means stride=kernel_size",
    args={
        "kernel_size": Arg(PAIR, min=1, help="Pooling window height and width; an int applies to both."),
        "stride": Arg(PAIR, 0, min=0, positional=False,
                      help="Step between windows on each axis; 0, the default, steps by one whole kernel_size."),
        "padding": Arg(PAIR, 0, min=0, positional=False,
                       help="Implicit -inf padding on each spatial side; at most kernel_size//2 per axis."),
    },
    validate=validate_padding,
    reference=_reference,
    examples=[
        Example("conv(8, kernel_size=3, padding=1)\nrelu()\nmax_pool(2)", ("B", 3, 16, 16), ("B", 8, 8, 8),
                "A 2x2 window with the default stride halves height and width."),
        Example("max_pool(2)\nflatten()\nlinear()", ("B", 4, 8, 8), ("B", 10),
                "Downsample, then classify the 4*4*4 remaining activations."),
        Example("max_pool(3, stride=2, padding=1)", ("B", 3, 15, 15), ("B", 3, 8, 8),
                "Overlapping 3x3 windows with stride 2 and padding 1."),
    ],
    category="spatial",
)
class MaxPool2d(nn.MaxPool2d):
    """Takes the maximum over each ``kernel_size`` window of the spatial axes
    of a ``[B, C, H, W]`` image, stepping by ``stride`` and treating the
    ``padding`` border as negative infinity so padded positions never win.
    Batch and channels pass through unchanged and

    ``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

    with the same rule on width; dilation is 1 and ``ceil_mode`` is false, so
    a trailing partial window is dropped. ``stride=0`` is the default and
    means "the same as ``kernel_size``", which is PyTorch's own default of
    non-overlapping windows; any other value is used as written. ``padding``
    may not exceed ``kernel_size//2`` on an axis. There are no parameters and
    no train/eval difference. The gradient routes to the argmax of each
    window, so ties send it to a single position.
    """

    def __init__(self, kernel_size, stride, padding):
        super().__init__(kernel_size, stride=normalize_stride(stride, kernel_size), padding=padding)
