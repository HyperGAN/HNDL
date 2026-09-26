from torch import nn
from torch.nn.utils import parametrizations

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, PAIR, Policy, operator
from ..relations import spatial


def _validate(args):
    if any(o >= s and o >= d for o, s, d in zip(args["output_padding"], args["stride"], args["dilation"])):
        raise HNDLError("E_ARGUMENT", "output_padding must be smaller than stride or dilation on each axis")


@operator(
    "deconv",
    identity="conv_transpose2d",
    summary="Transposed two-dimensional convolution, typically for upsampling.",
    shape="x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]",
    relation=spatial("conv_transpose2d"),
    shape_text="H_out = (H_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1, same for W",
    args={
        "out_channels": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL,
                            help="Output channels. Omit to infer from the consumer."),
        "in_channels": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Input channels. Normally inferred from the incoming tensor."),
        "kernel_size": Arg(PAIR, min=1, positional=False, help="Kernel height and width; an int applies to both."),
        "stride": Arg(PAIR, 1, min=1, positional=False, help="Upsampling step."),
        "padding": Arg(PAIR, 0, min=0, positional=False, help="Implicit zero padding removed from each side of the output."),
        "dilation": Arg(PAIR, 1, min=1, positional=False, help="Spacing between kernel taps."),
        "output_padding": Arg(PAIR, 0, min=0, positional=False, help="Extra size added to one side of each output axis."),
        "groups": Arg(int, 1, min=1, positional=False, help="Channel groups; must divide input and output channels."),
        "bias": Arg(bool, True, positional=False, help="Add a learned per-channel bias."),
        "spectral_norm": Arg(bool, False, positional=False,
                             help="Divide the weight by its largest singular value, estimated by power iteration."),
    },
    policies={"up2": Policy("spatial.up2_transpose@1", {"kernel_size": 4, "stride": 2, "padding": 1,
                                                         "dilation": 1, "output_padding": 0, "groups": 1})},
    validate=_validate,
    examples=[
        Example('deconv(64, policy="up2")\nrelu()\ndeconv(3, policy="up2")', ("B", 128, 4, 4), ("B", 3, 16, 16),
                'The "up2" policy selects kernel 4, stride 2, padding 1: exact doubling.'),
        Example("linear()\nreshape(32)\ndeconv(3, kernel_size=4, stride=2, padding=1)", ("B", 16), ("B", 3, 8, 8),
                "The seed 4×4 and projection width 512 are inferred backward."),
        Example('deconv(64, policy="up2", spectral_norm=True)\nrelu()\ndeconv(3, policy="up2")',
                ("B", 128, 8, 8), ("B", 3, 32, 32),
                "Spectral normalization also stabilizes a generator's upsampling stack."),
    ],
    category="convolution",
)
class ConvTranspose2d(nn.ConvTranspose2d):
    """The gradient of ``conv`` with respect to its input, used as a learned
    upsampling layer. Select ``policy="up2"`` to guarantee that height and
    width double; explicit arguments that contradict the policy fail.
    Parameters are ``weight`` with shape ``[in_channels, out_channels /
    groups, kH, kW]`` and, when ``bias=True``, ``bias`` with shape
    ``[out_channels]``.

    ## Spectral normalization

    With ``spectral_norm=True`` the weight is reparametrized as ``weight /
    sigma(weight)``, where ``sigma`` is the largest singular value estimated
    by one power iteration per forward pass
    (``torch.nn.utils.parametrizations.spectral_norm``). A transposed
    convolution stores its weight input-channel first, so the matrix view is
    taken over axis 1: the output-channel axis moves to the front and the
    rest flattens, exactly as PyTorch does for ``ConvTranspose2d``.

    The parametrization renames the registered state: the learned tensor
    becomes ``parametrizations.weight.original`` and ``weight`` turns into a
    computed attribute, with persistent buffers
    ``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
    holding the power-iteration vectors. ``init`` and ``trainable``
    overrides must therefore target ``parametrizations.weight.original``
    instead of ``weight``; ``bias`` is unaffected. The power iteration
    refreshes the buffers in training mode only, so evaluation is a pure
    function of the stored state.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, groups,
                 bias, spectral_norm):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        if spectral_norm:
            parametrizations.spectral_norm(self)
