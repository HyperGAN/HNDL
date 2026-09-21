from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator
from ._relations import conv_output


def _relation(s):
    args = s.args
    s.rank("x", 3)
    s.rank("out", 3)
    s.axis("x", 1, args.get("in_channels"))
    s.axis("out", 1, args.get("out_channels"))
    a, b = s.shape("x"), s.shape("out")
    s.arg("in_channels", a[1])
    s.arg("out_channels", b[1])
    for channels in (a[1], b[1]):
        if channels is not None and channels % args["groups"]:
            s.error("E_CONSTRAINT", f"groups={args['groups']} must divide input and output channels")
    kernel, stride, pad, dilation = (args[key] for key in ("kernel_size", "stride", "padding", "dilation"))
    length_in, length_out = a[2], b[2]
    if length_in is not None:
        s.axis("out", 2, conv_output(length_in, kernel, stride, pad, dilation))
    if length_out is not None:
        lower = max(1, (length_out - 1) * stride - 2 * pad + dilation * (kernel - 1) + 1)
        upper = length_out * stride - 2 * pad + dilation * (kernel - 1)
        if lower > upper:
            s.error("E_CONSTRAINT", "Convolution inverse has no positive input length")
        s.interval("x", 2, lower, upper)


def _reference(module):
    """The functional form of the same convolution, sharing the module's parameters.

    Under ``spectral_norm=True`` the weight is a parametrization that advances
    its power iteration on every training-mode access, so the weight is read in
    evaluation mode: that reuses the vectors the module's own forward just
    refreshed and therefore reproduces exactly the weight it applied.
    """
    def apply(x):
        training = module.training
        module.eval()
        try:
            weight = module.weight
        finally:
            module.train(training)
        return F.conv1d(x, weight, module.bias, stride=module.stride, padding=module.padding,
                        dilation=module.dilation, groups=module.groups)
    return apply


@operator(
    "conv1d",
    summary="One-dimensional convolution over [B, C, L] signals.",
    shape="x[B, C_in, L_in] -> out[B, C_out, L_out]",
    relation=_relation,
    shape_text="L_out = floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)",
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

    The input is `[B, C, L]`: channels at axis 1, positions at axis 2. The
    kernel slides along the length axis only. A `[B, T, D]` sequence carries
    its features on the last axis instead, so transpose it to `[B, D, T]`
    before this operator and back afterwards; `conv1d` never reinterprets the
    axes for you.

    With `groups > 1` the channels split into that many independent groups,
    each convolved with its own kernels; `groups == in_channels ==
    out_channels` is a depthwise convolution. Both channel counts must be
    divisible by `groups`.

    The output length is
    `floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)`.
    Because that formula is not injective, inferring the input length from a
    known output length generally leaves an interval of valid values; the
    resolver reports the ambiguity rather than choosing one.

    Parameters are `weight` of shape `[out_channels, in_channels / groups,
    kernel_size]` and, when enabled, `bias` of shape `[out_channels]`.
    Behavior is identical in train and eval mode unless `spectral_norm` is
    enabled, and the computation stays in the input dtype.

    ## Spectral normalization

    With `spectral_norm=True` the weight is reparametrized as `weight /
    sigma(weight)`, where `sigma` is the largest singular value of the weight
    viewed as an `[out_channels, -1]` matrix, estimated by one power iteration
    per forward pass
    (`torch.nn.utils.parametrizations.spectral_norm`). The layer is then
    1-Lipschitz, the standard constraint for a GAN discriminator.

    The parametrization renames the registered state: the learned tensor
    becomes `parametrizations.weight.original` and `weight` turns into a
    computed attribute, with persistent buffers
    `parametrizations.weight.0._u` and `parametrizations.weight.0._v` holding
    the power-iteration vectors. `init` and `trainable` overrides must
    therefore target `parametrizations.weight.original` instead of `weight`;
    `bias` is unaffected. The power iteration refreshes the buffers in
    training mode only, so evaluation is a pure function of the stored state.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                 spectral_norm):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
        if spectral_norm:
            parametrizations.spectral_norm(self)
