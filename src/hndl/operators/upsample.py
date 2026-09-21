from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, operator

MODES = ("nearest", "bilinear", "bicubic")


def _relation(s):
    scale = s.args["scale_factor"]
    s.rank("x", 4)
    s.rank("out", 4)
    a, b = s.shape("x"), s.shape("out")
    for axis in (2, 3):
        extent, target = a[axis], b[axis]
        if extent is not None:
            s.axis("out", axis, extent * scale)
        if target is not None:
            if target % scale:
                s.error("E_CONSTRAINT",
                        f"Target extent {target} on axis {axis} is not a multiple of scale_factor={scale}")
            s.axis("x", axis, target // scale)


def _validate(args):
    if args["mode"] == "nearest" and args["align_corners"]:
        raise HNDLError("E_ARGUMENT", "align_corners applies to bilinear and bicubic only; nearest ignores it")


def _reference(module):
    scale, mode, align_corners = int(module.scale_factor), module.mode, module.align_corners
    if mode == "nearest":
        return lambda x: x.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
    return lambda x: nn.functional.interpolate(x, scale_factor=float(scale), mode=mode, align_corners=align_corners)


@operator(
    "upsample",
    summary="Enlarge height and width by an integer factor with a fixed interpolation kernel.",
    shape="x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]",
    relation=_relation,
    shape_text="H_out = H_in * scale_factor, W_out = W_in * scale_factor; channels unchanged",
    args={
        "scale_factor": Arg(int, 2, min=1, help="Integer factor applied to height and width."),
        "mode": Arg(str, "nearest", choices=MODES, positional=False,
                    help="Interpolation kernel: nearest, bilinear, or bicubic."),
        "align_corners": Arg(bool, False, positional=False,
                             help="Align the corner pixels of input and output; bilinear and bicubic only."),
    },
    validate=_validate,
    reference=_reference,
    examples=[
        Example("upsample(2)\nconv(3, kernel_size=3, padding=1)", ("B", 3, 8, 8), ("B", 3, 16, 16),
                "Nearest-neighbour doubling followed by a convolution that smooths the blocks."),
        Example('upsample(2, mode="bilinear")', ("B", 4, 8, 8), ("B", 4, 16, 16),
                "Bilinear doubling with align_corners=False, the PyTorch default."),
        Example("conv(8, kernel_size=3, padding=1)\nupsample(4)", ("B", 3, 4, 4), ("B", 8, 16, 16),
                "A single stage that quadruples both spatial axes."),
        Example("linear()\nreshape(16)\nupsample(2)", ("B", 32), ("B", 16, 8, 8),
                "The 4x4 seed and the projection width 256 are inferred backward through the upsampling."),
    ],
    category="spatial",
)
class Upsample(nn.Upsample):
    """Resamples the spatial axes of a ``[B, C, H, W]`` image by an exact
    integer factor: ``H_out = H * scale_factor`` and
    ``W_out = W * scale_factor``. The channel axis is untouched and there are
    no parameters, so train and eval behave identically and the layer is a
    pure function of its input.

    Modes:

    - ``nearest`` repeats each input pixel in a ``scale_factor`` x
      ``scale_factor`` block. It is the cheapest choice and is exactly
      equivalent to ``x.repeat_interleave(scale_factor, 2).repeat_interleave(scale_factor, 3)``.
    - ``bilinear`` and ``bicubic`` interpolate between neighbouring pixels.
      ``align_corners`` selects the sampling grid convention: ``False`` (the
      default) treats pixels as areas, ``True`` pins the corner pixel centers
      of input and output together.

    ``align_corners`` is meaningful only for ``bilinear`` and ``bicubic``;
    with ``nearest`` it must stay ``False``, and the underlying PyTorch call
    receives ``None`` so no warning is emitted. Note that ``bicubic`` can
    overshoot the input range; clamp afterwards if bounded output matters.

    Resolution is bidirectional: a known input extent fixes the output, and a
    known output extent fixes the input when it divides by ``scale_factor``,
    otherwise ``E_CONSTRAINT`` is reported.
    """

    def __init__(self, scale_factor, mode, align_corners):
        super().__init__(scale_factor=float(scale_factor), mode=mode,
                         align_corners=None if mode == "nearest" else align_corners)
