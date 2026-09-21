from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    factor = s.args["upscale_factor"]
    block = factor * factor
    s.rank("x", 4)
    s.rank("out", 4)
    a, b = s.shape("x"), s.shape("out")
    if a[1] is not None:
        if a[1] % block:
            s.error("E_CONSTRAINT",
                    f"Input channels {a[1]} are not divisible by upscale_factor**2 = {block}")
        s.axis("out", 1, a[1] // block)
    if b[1] is not None:
        s.axis("x", 1, b[1] * block)
    for axis in (2, 3):
        extent, target = a[axis], b[axis]
        if extent is not None:
            s.axis("out", axis, extent * factor)
        if target is not None:
            if target % factor:
                s.error("E_CONSTRAINT",
                        f"Target extent {target} on axis {axis} is not a multiple of upscale_factor={factor}")
            s.axis("x", axis, target // factor)


def _reference(module):
    factor = module.upscale_factor

    def call(x):
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch, channels // (factor * factor), factor, factor, height, width)
        return grouped.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, channels // (factor * factor), height * factor, width * factor)

    return call


@operator(
    "pixel_shuffle",
    summary="Trade channels for resolution: rearrange C*r^2 channels into an r-times larger image.",
    shape="x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]",
    relation=_relation,
    shape_text="C_in == C_out * upscale_factor**2; H_out = H_in * upscale_factor, W_out = W_in * upscale_factor",
    args={
        "upscale_factor": Arg(int, min=1, help="Factor r: consumes r^2 channels per output channel."),
    },
    reference=_reference,
    examples=[
        Example("conv(48, kernel_size=3, padding=1)\npixel_shuffle(4)", ("B", 16, 8, 8), ("B", 3, 32, 32),
                "The sub-pixel convolution of ESPCN: 48 = 3 * 4^2 channels become a 4x larger RGB image."),
        Example("pixel_shuffle(2)", ("B", 12, 4, 4), ("B", 3, 8, 8),
                "A bare shuffle: 12 channels at 4x4 become 3 channels at 8x8."),
        Example("conv(kernel_size=3, padding=1)\npixel_shuffle(2)", ("B", 3, 8, 8), ("B", 5, 16, 16),
                "The convolution width 20 is inferred backward from the shuffled output."),
    ],
    category="spatial",
)
class PixelShuffle(nn.PixelShuffle):
    """Rearranges a ``[B, C*r^2, H, W]`` tensor into ``[B, C, H*r, W*r]``,
    where ``r`` is ``upscale_factor``. Channel ``c*r^2 + i*r + j`` of the
    input supplies output channel ``c`` at the sub-pixel offset ``(i, j)``
    inside each ``r`` x ``r`` output block:

    ```text
    out[b, c, h*r + i, w*r + j] == x[b, c*r^2 + i*r + j, h, w]
    ```

    This is the standard sub-pixel convolution upsampler: a preceding
    ``conv`` produces ``r^2`` values per output pixel and the shuffle lays
    them out spatially, which avoids the checkerboard artifacts of a strided
    transposed convolution. It is a pure permutation and reshape, with no
    parameters, no arithmetic, and identical train and eval behavior; the
    output dtype is the input dtype.

    Input channels must be divisible by ``upscale_factor**2`` and a known
    output extent must be divisible by ``upscale_factor``; otherwise
    ``E_CONSTRAINT`` is reported. Resolution runs in both directions, so the
    channel width of the producing layer can be left to the solver.
    """

    def __init__(self, upscale_factor):
        super().__init__(upscale_factor)
