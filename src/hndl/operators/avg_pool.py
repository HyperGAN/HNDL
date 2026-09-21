import torch.nn.functional as F
from torch import nn

from ..operator import Arg, Example, PAIR, operator
from ._pooling import normalize_stride, pooling, validate_padding


def _reference(module):
    def forward(x):
        return F.avg_pool2d(x, module.kernel_size, stride=module.stride, padding=module.padding,
                            count_include_pad=module.count_include_pad)
    return forward


@operator(
    "avg_pool",
    identity="avg_pool2d",
    summary="Two-dimensional average pooling over [B, C, H, W] images.",
    shape="x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]",
    relation=pooling(),
    shape_text="H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1, same for W; stride=0 means stride=kernel_size",
    args={
        "kernel_size": Arg(PAIR, min=1, help="Pooling window height and width; an int applies to both."),
        "stride": Arg(PAIR, 0, min=0, positional=False,
                      help="Step between windows on each axis; 0, the default, steps by one whole kernel_size."),
        "padding": Arg(PAIR, 0, min=0, positional=False,
                       help="Implicit zero padding on each spatial side; at most kernel_size//2 per axis."),
        "count_include_pad": Arg(bool, True, positional=False,
                                 help="Divide padded windows by the full window area instead of their real elements."),
    },
    validate=validate_padding,
    reference=_reference,
    examples=[
        Example("conv(8, kernel_size=3, padding=1)\nrelu()\navg_pool(2)", ("B", 3, 16, 16), ("B", 8, 8, 8),
                "A 2x2 window with the default stride halves height and width."),
        Example("avg_pool(4)\nflatten()\nlinear()", ("B", 8, 4, 4), ("B", 5),
                "A window covering the whole image is global average pooling."),
        Example("avg_pool(3, stride=2, padding=1, count_include_pad=False)", ("B", 3, 15, 15), ("B", 3, 8, 8),
                "Overlapping windows where the border averages only real elements."),
    ],
    category="spatial",
)
class AvgPool2d(nn.AvgPool2d):
    """Averages each ``kernel_size`` window of the spatial axes of a
    ``[B, C, H, W]`` image, stepping by ``stride``. Batch and channels pass
    through unchanged and

    ``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

    with the same rule on width; ``ceil_mode`` is false, so a trailing
    partial window is dropped. ``stride=0`` is the default and means "the
    same as ``kernel_size``", which is PyTorch's own default of
    non-overlapping windows; any other value is used as written. ``padding``
    adds zeros and may not exceed ``kernel_size//2`` on an axis: with
    ``count_include_pad`` true, the default, those zeros count in the
    denominator, and with it false each window divides by the number of real
    input elements it covers. There are no parameters and no train/eval
    difference; the gradient is spread equally over every counted position.
    The sum accumulates in the input dtype, so a large window in float16
    loses precision.
    """

    def __init__(self, kernel_size, stride, padding, count_include_pad):
        super().__init__(kernel_size, stride=normalize_stride(stride, kernel_size), padding=padding,
                         count_include_pad=count_include_pad)
