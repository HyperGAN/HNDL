from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator
from ..relations import conv_axis
from ._equalized import eval_weight


def _validate(args):
    if args.get("causal") and args.get("padding", 0) != 0:
        raise HNDLError("E_ARGUMENT", "causal=True requires padding=0")


def _relation(s):
    args = s.args
    channel_axis = 2 if args.get("channels_last") else 1
    length_axis = 1 if args.get("channels_last") else 2
    s.rank("x", 3)
    s.rank("out", 3)
    s.axis("x", channel_axis, args.get("in_channels"))
    s.axis("out", channel_axis, args.get("out_channels"))
    a, b = s.shape("x"), s.shape("out")
    s.arg("in_channels", a[channel_axis])
    s.arg("out_channels", b[channel_axis])
    for channels in (a[channel_axis], b[channel_axis]):
        if channels is not None and channels % args["groups"]:
            s.error("E_CONSTRAINT", f"groups={args['groups']} must divide input and output channels")

    if args.get("causal"):
        stride = args["stride"]
        in_len = a[length_axis]
        out_len = b[length_axis]
        if stride == 1:
            if in_len is not None:
                s.axis("out", length_axis, in_len)
            if out_len is not None:
                s.axis("x", length_axis, out_len)
        else:
            if in_len is not None:
                s.axis("out", length_axis, (in_len - 1) // stride + 1)
            elif out_len is not None:
                s.error("E_AMBIGUOUS", f"Causal conv1d with stride={stride} cannot uniquely infer input length from output length {out_len}")
    else:
        conv_axis(s, length_axis, *(args[key] for key in ("kernel_size", "stride", "padding", "dilation")))


def _reference(module):
    """The functional form of the same convolution, sharing the module's parameters.

    Under ``spectral_norm=True`` the weight is a parametrization that advances
    its power iteration on every training-mode access, so the weight is read in
    evaluation mode: that reuses the vectors the module's own forward just
    refreshed and therefore reproduces exactly the weight it applied.
    """
    def apply(x):
        if module.channels_last:
            x = x.transpose(1, 2)
        if module.causal:
            left_pad = module.dilation[0] * (module.kernel_size[0] - 1)
            x = F.pad(x, (left_pad, 0))
            padding = 0
        else:
            padding = module.padding
        weight = eval_weight(module)
        out = F.conv1d(x, weight, module.bias, stride=module.stride, padding=padding,
                       dilation=module.dilation, groups=module.groups)
        if module.channels_last:
            out = out.transpose(1, 2).contiguous()
        return out
    return apply


@operator(
    "conv1d",
    summary="One-dimensional convolution over [B, C, L] or [B, T, C] signals.",
    shape="x -> out",
    relation=_relation,
    validate=_validate,
    shape_text="x[B, C_in, L_in] -> out[B, C_out, L_out] (or x[B, L_in, C_in] -> out[B, L_out, C_out] when channels_last=True)",
    reference=_reference,
    args={
        "out_channels": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL,
                            help="Output channels. Omit to infer from the consumer."),
        "in_channels": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Input channels. Normally inferred from the incoming tensor."),
        "kernel_size": Arg(int, min=1, positional=False, help="Kernel width in positions along the length axis."),
        "stride": Arg(int, 1, min=1, positional=False, help="Step between kernel applications along the length axis."),
        "padding": Arg(int, 0, min=0, positional=False, help="Zero padding added to each end of the length axis."),
        "dilation": Arg(int, 1, min=1, positional=False, help="Spacing between kernel taps along the length axis."),
        "groups": Arg(int, 1, min=1, positional=False, help="Channel groups; must divide input and output channels."),
        "bias": Arg(bool, True, positional=False, help="Add a learned per-channel bias."),
        "spectral_norm": Arg(bool, False, positional=False,
                             help="Divide the weight by its largest singular value, estimated by power iteration."),
        "causal": Arg(bool, False, positional=False, since="0.9.0",
                      help="Left-only causal padding; when stride=1 establishes bijective L_out == L_in."),
        "channels_last": Arg(bool, False, positional=False, since="0.9.0",
                             help="Sequence layout [B, T, C] instead of [B, C, L]."),
    },
    examples=[
        Example("conv1d(16, kernel_size=3, padding=1)\nrelu()\nconv1d(4, kernel_size=3, padding=1)",
                ("B", 4, 32), ("B", 4, 32), "Padding 1 with kernel 3 preserves the length axis."),
        Example("conv1d(8, kernel_size=4, stride=2, padding=1)", ("B", 3, 16), ("B", 8, 8),
                "Kernel 4, stride 2, padding 1 halves the length."),
        Example("conv1d(6, kernel_size=3, dilation=2)\ngroup_norm(3)\nrelu()", ("B", 2, 16), ("B", 6, 12),
                "Dilation 2 widens the receptive field to 5 positions and trims 4 from the length."),
        Example("conv1d(16, kernel_size=4, stride=2, padding=1, spectral_norm=True)\nleaky_relu(0.2)\n"
                "conv1d(32, kernel_size=4, stride=2, padding=1, spectral_norm=True)",
                ("B", 1, 64), ("B", 32, 16),
                "A spectrally normalized waveform discriminator trunk."),
        Example("conv1d(32, kernel_size=4, causal=True)", ("B", 16, 64), ("B", 32, 64),
                "Causal convolution preserves sequence length bijectively when stride=1."),
        Example("conv1d(32, kernel_size=4, causal=True, channels_last=True)", ("B", 64, 16), ("B", 64, 32),
                "Causal convolution directly on [B, T, C] sequence layout."),
    ],
    category="convolution",
)
class Conv1d(nn.Conv1d):
    """Cross-correlation of the input with `out_channels` learned kernels of
    shape `[in_channels / groups, kernel_size]`:

    ```text
    out[b, co, l] = bias[co] + sum_{ci, k} weight[co, ci, k] * x[b, ci, l*stride - padding + k*dilation]
    ```

    ## Axis convention

    The default input is `[B, C, L]`: channels at axis 1, positions at axis 2.
    With `channels_last=True`, the input layout is `[B, T, C]`.
    With `causal=True`, left-padding of `dilation * (kernel_size - 1)` is applied
    so each position attends only to past and present positions.
    When `causal=True` and `stride=1`, `L_out == L_in` forms a bijective relation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                 spectral_norm, causal=False, channels_last=False):
        self.causal = bool(causal)
        self.channels_last = bool(channels_last)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
        if spectral_norm:
            parametrizations.spectral_norm(self)

    def forward(self, x):
        if self.channels_last:
            x = x.transpose(1, 2)
        if self.causal:
            left_pad = self.dilation[0] * (self.kernel_size[0] - 1)
            x = F.pad(x, (left_pad, 0))
        out = super().forward(x)
        if self.channels_last:
            out = out.transpose(1, 2).contiguous()
        return out
