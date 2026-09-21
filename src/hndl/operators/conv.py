from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, PAIR, operator
from ._relations import spatial


@operator(
    "conv",
    identity="conv2d",
    summary="Two-dimensional convolution over [B, C, H, W] images.",
    shape="x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]",
    relation=spatial("conv2d"),
    shape_text="H_out = floor((H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1), same for W",
    args={
        "out_channels": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL,
                            help="Output channels. Omit to infer from the consumer."),
        "in_channels": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Input channels. Normally inferred from the incoming tensor."),
        "kernel_size": Arg(PAIR, min=1, positional=False, help="Kernel height and width; an int applies to both."),
        "stride": Arg(PAIR, 1, min=1, positional=False, help="Step between kernel applications."),
        "padding": Arg(PAIR, 0, min=0, positional=False, help="Zero padding added to each spatial side."),
        "dilation": Arg(PAIR, 1, min=1, positional=False, help="Spacing between kernel taps."),
        "groups": Arg(int, 1, min=1, positional=False, help="Channel groups; must divide input and output channels."),
        "bias": Arg(bool, True, positional=False, help="Add a learned per-channel bias."),
    },
    examples=[
        Example("conv(16, kernel_size=3, padding=1)\nrelu()\nconv(3, kernel_size=3, padding=1)",
                ("B", 3, 32, 32), ("B", 3, 32, 32), "Padding 1 with kernel 3 preserves height and width."),
        Example("conv(8, kernel_size=4, stride=2, padding=1)", ("B", 3, 16, 16), ("B", 8, 8, 8),
                "Kernel 4, stride 2, padding 1 halves each spatial axis."),
    ],
    category="convolution",
)
class Conv2d(nn.Conv2d):
    """Cross-correlation of the input with ``out_channels`` learned kernels of
    shape ``[in_channels / groups, kH, kW]``. Inverse inference from a known
    output extent may leave an interval of valid input sizes; that ambiguity
    is reported rather than resolved arbitrarily.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
