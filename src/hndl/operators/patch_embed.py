import torch.nn.functional as F
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    s.rank("x", 4)
    s.rank("out", 3)
    patch = s.args["patch_size"]
    height, width = s.shape("x")[2], s.shape("x")[3]
    for extent, label in ((height, "height"), (width, "width")):
        if extent is not None and extent % patch:
            s.error("E_CONSTRAINT", f"patch_size={patch} must divide the input {label}={extent}")
    positions = s.shape("out")[1]
    if height is not None and width is not None:
        s.axis("out", 1, (height // patch) * (width // patch))
    elif positions is not None and (height is not None or width is not None):
        # One spatial extent plus the token count fixes the other; with neither
        # known the factorization is ambiguous and is left to the solver to report.
        axis, known = (3, height) if height is not None else (2, width)
        tiles = known // patch
        if positions % tiles:
            s.error("E_CONSTRAINT",
                    f"{positions} tokens cannot be split into {tiles} rows or columns of patches")
        s.axis("x", axis, (positions // tiles) * patch)


def _reference(module):
    weight = module.proj.weight
    bias = module.proj.bias
    stride = module.proj.stride

    def run(x):
        return F.conv2d(x, weight, bias, stride=stride).flatten(2).transpose(1, 2)

    return run


@operator(
    "patch_embed",
    summary="Cut an image into non-overlapping patches and embed each one as a token.",
    shape="x[B, C, H, W] -> out[B, N, D]",
    relation=_relation,
    reference=_reference,
    shape_text="N == (H / patch_size) * (W / patch_size); patch_size must divide H and W; D == dim",
    args={
        "dim": Arg(int, inferable=True, dim="D", min=1, max=MAX_DIMENSION_LITERAL,
                   help="Token embedding width. Omit it to infer the width from what follows."),
        "patch_size": Arg(int, min=1, max=MAX_DIMENSION_LITERAL,
                          help="Side of the square patch; it must divide both the height and the width."),
        "in_channels": Arg(int, inferable=True, dim="C", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Image channels at axis 1. Normally inferred from the incoming tensor."),
    },
    examples=[
        Example("patch_embed(32, 4)", ("B", 3, 16, 16), ("B", 16, 32),
                "A 16x16 image becomes 4x4 = 16 tokens of width 32."),
        Example("patch_embed(16, 8)\nlinear()", ("B", 3, 16, 16), ("B", 4, 8),
                "Four patch tokens feed an ordinary linear map over the token axis."),
        Example("patch_embed(patch_size=2)\nrelu()", ("B", 3, 8, 8), ("B", 16, 64),
                "The embedding width is inferred backward from the output contract."),
    ],
    category="vision",
)
class PatchEmbed(nn.Module):
    """The patch-embedding stem of a vision transformer.

    The input image ``[B, C, H, W]`` is cut into non-overlapping ``patch_size``
    x ``patch_size`` tiles and each tile is projected to ``dim`` features:

    ```
    out = proj(x).flatten(2).transpose(1, 2)
    ```

    where ``proj`` is an ``nn.Conv2d(in_channels, dim, kernel_size=patch_size,
    stride=patch_size)``. Because the kernel and the stride are equal, the
    convolution reads every pixel exactly once, so the layer is a per-patch
    affine map on the flattened ``[C, patch_size, patch_size]`` window.

    Axis conventions: the input is an image ``[B, C, H, W]`` and the output is
    a sequence ``[B, N, D]`` with ``N = (H / patch_size) * (W / patch_size)``
    tokens in row-major order (all patches of the first patch row first) and
    ``D = dim`` features on the last axis, which is what the sequence
    operations such as ``linear`` act on. ``patch_size`` must divide both ``H``
    and ``W``; a remainder is reported as ``E_CONSTRAINT`` rather than being
    cropped away. Inference runs in both directions: ``dim`` and
    ``in_channels`` follow the neighbouring contracts, and a known token count
    fixes the missing spatial extent when the other one is known. A token
    count alone does not determine ``H`` and ``W`` and is reported as
    ambiguous instead of guessed.

    No positional embedding, class token, or normalization is added; compose
    those explicitly. Behavior is identical in train and eval mode. The single
    submodule is ``proj``, whose parameters are ``proj.weight`` of shape
    ``[dim, in_channels, patch_size, patch_size]`` and ``proj.bias``. The
    projection runs entirely in the plan compute dtype.
    """

    def __init__(self, dim, patch_size, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
