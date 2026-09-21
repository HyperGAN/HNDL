from torch import nn
from torch.nn.utils import parametrizations

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, PAIR, Policy, operator
from ._relations import spatial

DOWN2 = "spatial.down2@1"
_spatial = spatial("conv2d")


def _relation(s):
    """Convolution arithmetic, plus the exact halving the "down2" policy promises.

    A kernel 4 / stride 2 / padding 1 convolution maps both ``2*O`` and
    ``2*O + 1`` to ``O``, so the generic inverse leaves a two-value interval.
    Under ``policy="down2"`` an odd input extent is rejected and each spatial
    axis is pinned to ``H_in = 2*H_out``, which makes backward inference unique.
    """
    _spatial(s)
    if s.policy != DOWN2:
        return
    x, out = s.shape("x"), s.shape("out")
    if x is None or out is None:
        return
    for axis in (2, 3):
        extent, target = x[axis], out[axis]
        if extent is not None:
            if extent % 2:
                s.error("E_CONSTRAINT",
                        f'policy="down2" halves each spatial axis, but input axis {axis} has odd extent {extent}')
            s.axis("out", axis, extent // 2)
        if target is not None:
            s.axis("x", axis, 2 * target)


@operator(
    "conv",
    identity="conv2d",
    summary="Two-dimensional convolution over [B, C, H, W] images.",
    shape="x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]",
    relation=_relation,
    shape_text=("H_out = floor((H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1), same for W; "
                'policy="down2" additionally requires even input extents and H_out = H_in/2, W_out = W_in/2'),
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
        "spectral_norm": Arg(bool, False, positional=False,
                             help="Divide the weight by its largest singular value, estimated by power iteration."),
    },
    policies={"down2": Policy(DOWN2, {"kernel_size": 4, "stride": 2, "padding": 1, "dilation": 1, "groups": 1})},
    examples=[
        Example("conv(16, kernel_size=3, padding=1)\nrelu()\nconv(3, kernel_size=3, padding=1)",
                ("B", 3, 32, 32), ("B", 3, 32, 32), "Padding 1 with kernel 3 preserves height and width."),
        Example("conv(8, kernel_size=4, stride=2, padding=1)", ("B", 3, 16, 16), ("B", 8, 8, 8),
                "Kernel 4, stride 2, padding 1 halves each spatial axis."),
        Example('conv(64, policy="down2")\nleaky_relu(0.2)\nconv(128, policy="down2")\nflatten()\nlinear()',
                ("B", 3, 32, 32), ("B", 1),
                'A DCGAN-style discriminator: the "down2" policy fixes kernel 4, stride 2, padding 1, and the '
                "8x8 feature map and its 8192-wide flattening resolve backward."),
        Example('conv(64, policy="down2", spectral_norm=True)\nleaky_relu(0.2)\n'
                'conv(128, policy="down2", spectral_norm=True)\nleaky_relu(0.2)\nflatten()\n'
                "linear(spectral_norm=True)",
                ("B", 3, 32, 32), ("B", 1),
                "An SN-GAN discriminator: the same downsampling stack with every weight constrained to unit "
                "spectral norm."),
    ],
    category="convolution",
)
class Conv2d(nn.Conv2d):
    """Cross-correlation of the input with ``out_channels`` learned kernels of
    shape ``[in_channels / groups, kH, kW]``. The input is ``[B, C_in, H_in,
    W_in]`` and the output ``[B, C_out, H_out, W_out]``, with

    ```text
    H_out = floor((H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)
    ```

    and the same formula on the width axis. Parameters are ``weight`` with
    shape ``[out_channels, in_channels / groups, kH, kW]`` and, when
    ``bias=True``, ``bias`` with shape ``[out_channels]``. Behavior is
    identical in training and evaluation unless ``spectral_norm`` is enabled.

    Inverse inference from a known output extent may leave an interval of
    valid input sizes; that ambiguity is reported rather than resolved
    arbitrarily. Select ``policy="down2"`` for the standard downsampling
    block — kernel 4, stride 2, padding 1, dilation 1, groups 1 — which also
    *asserts* exact halving: input extents must be even, ``H_out = H_in / 2``
    and ``W_out = W_in / 2``. That removes the usual off-by-one interval, so a
    stack of ``down2`` convolutions resolves backward from the output contract
    alone. An explicit argument contradicting the policy fails with
    ``E_POLICY_CONFLICT``; an odd input extent under the policy fails with
    ``E_CONSTRAINT``.

    ## Spectral normalization

    With ``spectral_norm=True`` the weight is reparametrized as ``weight /
    sigma(weight)``, where ``sigma`` is the largest singular value of the
    weight viewed as an ``[out_channels, -1]`` matrix, estimated by one power
    iteration per forward pass
    (``torch.nn.utils.parametrizations.spectral_norm``). This is the
    discriminator constraint from SN-GAN; pair it with ``policy="down2"`` for
    the usual downsampling critic.

    The parametrization renames the registered state: the learned tensor
    becomes ``parametrizations.weight.original`` and ``weight`` turns into a
    computed attribute, with persistent buffers
    ``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
    holding the power-iteration vectors. ``init`` and ``trainable``
    overrides must therefore target ``parametrizations.weight.original``
    instead of ``weight``; ``bias`` is unaffected. The power iteration
    refreshes the buffers in training mode only, so evaluation is a pure
    function of the stored state — the one case where this operator's
    behavior differs between training and evaluation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                 spectral_norm):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
        if spectral_norm:
            parametrizations.spectral_norm(self)
