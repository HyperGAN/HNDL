from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    mode, padding_mode, align_corners = module.mode, module.padding_mode, module.align_corners

    def grid_sample(x, grid):
        return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return grid_sample


@operator(
    "grid_sample",
    summary="Sample an image using a per-example normalized coordinate grid.",
    shape="x[B, C, H, W], grid[B, OH, OW, 2] -> out[B, C, OH, OW]",
    args={
        "mode": Arg(str, "bilinear", choices=("bilinear", "nearest", "bicubic"),
                    positional=False,
                    help="Interpolation kernel."),
        "padding_mode": Arg(str, "zeros", choices=("zeros", "border", "reflection"),
                            positional=False,
                            help="How samples outside the input are filled."),
        "align_corners": Arg(bool, False, positional=False,
                             help="Whether -1 and 1 refer to corner pixel centers."),
    },
    reference=_reference,
    examples=[Example('g = coordinate_grid(x, 8, 8)\ngrid_sample(x, g)',
                      ("B", 3, 4, 4), ("B", 3, 8, 8), "Sample a 4x4 image on an 8x8 grid.")],
    category="spatial",
)
class GridSample(nn.Module):
    """A direct ``torch.nn.functional.grid_sample`` with an explicit grid.

    The grid's last axis is (x,y) in normalized [-1,1] coordinates. Image
    and grid batches must match. Defaults are bilinear interpolation, zero
    padding, and align_corners=False. No parameters or random draws;
    train and eval are identical. Gradients flow to image and coordinates
    as supported by PyTorch. CUDA backward can be nondeterministic, and
    PyTorch's sampler does not support double backward.
    """

    def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=False):
        super().__init__()
        self.mode, self.padding_mode, self.align_corners = mode, padding_mode, align_corners

    def forward(self, x, grid):
        return F.grid_sample(x, grid, mode=self.mode, padding_mode=self.padding_mode,
                             align_corners=self.align_corners)

    def extra_repr(self):
        return f"mode={self.mode!r}, padding_mode={self.padding_mode!r}, align_corners={self.align_corners}"
