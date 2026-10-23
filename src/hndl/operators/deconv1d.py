"""One-dimensional transposed convolution for waveform synthesis and upsampling."""

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, Policy, operator
from ..relations import conv_transpose_axis
from ._equalized import eval_weight


def _validate(args):
    if args["output_padding"] >= args["stride"] and args["output_padding"] >= args["dilation"]:
        raise HNDLError("E_ARGUMENT", "output_padding must be smaller than stride or dilation")
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
        if in_len is not None:
            s.axis("out", length_axis, in_len * stride)
        if out_len is not None:
            if out_len % stride != 0:
                s.error("E_CONSTRAINT", f"Causal deconv1d output length {out_len} must be divisible by stride {stride}")
            else:
                s.axis("x", length_axis, out_len // stride)
    elif all(args.get(key) is not None for key in ("kernel_size", "stride", "padding", "dilation", "output_padding")):
        conv_transpose_axis(
            s,
            length_axis,
            kernel=args["kernel_size"],
            stride=args["stride"],
            padding=args["padding"],
            dilation=args["dilation"],
            output_padding=args["output_padding"],
        )


def _reference(module):
    """The functional form of the 1D transposed convolution."""

    def apply(x):
        channels_last = getattr(module, "channels_last", False)
        causal = getattr(module, "causal", False)
        if channels_last:
            x = x.transpose(1, 2)
        weight = eval_weight(module)
        out = F.conv_transpose1d(
            x,
            weight,
            module.bias,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            groups=module.groups,
            dilation=module.dilation,
        )
        if causal:
            target_len = x.shape[-1] * module.stride[0]
            out = out[..., :target_len]
        if channels_last:
            out = out.transpose(1, 2).contiguous()
        return out

    return apply


@operator(
    "deconv1d",
    identity="conv_transpose1d",
    summary="Transposed one-dimensional convolution for waveform synthesis and upsampling.",
    shape="x[B, C_in, L_in] -> out[B, C_out, L_out]",
    relation=_relation,
    reference=_reference,
    shape_text="L_out = (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1",
    args={
        "out_channels": Arg(
            int,
            inferable=True,
            min=1,
            max=MAX_DIMENSION_LITERAL,
            help="Output channels. Omit to infer from the consumer.",
        ),
        "in_channels": Arg(
            int,
            inferable=True,
            min=1,
            max=MAX_DIMENSION_LITERAL,
            positional=False,
            help="Input channels. Normally inferred from the incoming tensor.",
        ),
        "kernel_size": Arg(
            int, min=1, help="Kernel length in positions along the length axis."
        ),
        "stride": Arg(int, 1, min=1, positional=False, help="Upsampling step along the length axis."),
        "padding": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="Implicit zero padding removed from each side of the output.",
        ),
        "dilation": Arg(int, 1, min=1, positional=False, help="Spacing between kernel taps."),
        "output_padding": Arg(
            int,
            0,
            min=0,
            positional=False,
            help="Extra size added to one side of the output length axis.",
        ),
        "groups": Arg(
            int,
            1,
            min=1,
            positional=False,
            help="Channel groups; must divide input and output channels.",
        ),
        "bias": Arg(bool, True, positional=False, help="Add a learned per-channel bias."),
        "spectral_norm": Arg(
            bool,
            False,
            positional=False,
            help="Divide the weight by its largest singular value, estimated by power iteration.",
        ),
        "causal": Arg(
            bool,
            False,
            positional=False,
            since="0.10.0",
            help="Left-only causal padding ensuring output length is L_in * stride.",
        ),
        "channels_last": Arg(
            bool,
            False,
            positional=False,
            since="0.10.0",
            help="Layout [B, T, C] matching conv1d.",
        ),
    },
    policies={
        "up2": Policy(
            "spatial1d.up2_transpose@1",
            {
                "kernel_size": 4,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "output_padding": 0,
                "groups": 1,
            },
        )
    },
    validate=_validate,
    examples=[
        Example(
            'deconv1d(64, policy="up2")\nrelu()\ndeconv1d(1, policy="up2")',
            ("B", 128, 16),
            ("B", 1, 64),
            'The "up2" policy selects kernel 4, stride 2, padding 1: exact length doubling.',
        ),
        Example(
            "deconv1d(16, kernel_size=4, stride=2, padding=1)",
            ("B", 8, 32),
            ("B", 16, 64),
            "Explicit kernel parameters double the length axis.",
        ),
        Example(
            'deconv1d(32, policy="up2", spectral_norm=True)',
            ("B", 16, 8),
            ("B", 32, 16),
            "Spectrally normalized 1D upsampling layer for audio generators.",
        ),
    ],
    category="convolution",
)
class ConvTranspose1d(nn.ConvTranspose1d):
    """The gradient of ``conv1d`` with respect to its input, used as a learned
    upsampling layer along the length axis of ``[B, C, L]`` signals.

    Select ``policy="up2"`` to guarantee that length doubles exactly.
    Parameters are ``weight`` of shape ``[in_channels, out_channels / groups, kernel_size]``
    and, when ``bias=True``, ``bias`` of shape ``[out_channels]``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        bias,
        spectral_norm,
        causal=False,
        channels_last=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.causal = bool(causal)
        self.channels_last = bool(channels_last)
        if spectral_norm:
            parametrizations.spectral_norm(self)

    def forward(self, x):
        if self.channels_last:
            x = x.transpose(1, 2)
        out = super().forward(x)
        if self.causal:
            target_len = x.shape[-1] * self.stride[0]
            out = out[..., :target_len]
        if self.channels_last:
            out = out.transpose(1, 2).contiguous()
        return out

    def extra_repr(self):
        s = super().extra_repr()
        if self.causal:
            s += ", causal=True"
        if self.channels_last:
            s += ", channels_last=True"
        return s
