import torch
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


@operator(
    "coordinate_grid",
    summary="Emit a normalized 2D coordinate grid in (x, y) order.",
    shape="x[B, ...]:any -> out[B, H, W, 2]",
    args={
        "height": Arg(int, dim="H", min=2, max=MAX_DIMENSION_LITERAL,
                      help="Grid rows, with centers spanning -1 to 1 inclusive."),
        "width": Arg(int, dim="W", min=2, max=MAX_DIMENSION_LITERAL,
                     help="Grid columns, with centers spanning -1 to 1 inclusive."),
    },
    examples=[Example('coordinate_grid(4, 6)', ("B", 8), ("B", 4, 6, 2),
                      "A fixed rectangular coordinate grid; only the input batch size is used.",
                      input_dtype="int64")],
    category="spatial",
)
class CoordinateGrid(nn.Module):
    """Emit ``[B,H,W,2]`` coordinates spanning [-1,1] including both endpoints.

    The final axis is (x, y); x varies along columns, y along rows. Input
    values are ignored: only batch size is used. The fixed ``grid`` buffer
    follows model device/dtype and is saved in the state dict. No random
    draws or trainable parameters. There is no gradient to the input.
    This is the endpoint convention of ``linspace(-1,1,extent)``, regardless
    of the ``align_corners`` choice in a downstream sampler.
    """

    def __init__(self, height, width):
        super().__init__()
        y, x = torch.meshgrid(torch.linspace(-1, 1, height),
                              torch.linspace(-1, 1, width), indexing="ij")
        self.register_buffer("grid", torch.stack((x, y), dim=-1).unsqueeze(0))

    def forward(self, x):
        return self.grid.expand(x.shape[0], -1, -1, -1)
