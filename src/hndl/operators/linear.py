from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


@operator(
    "linear",
    summary="Fully connected layer: a learned affine map on the feature axis.",
    shape="x[B, D_in] -> out[B, D_out]",
    args={
        "out_features": Arg(int, inferable=True, dim="D_out", min=1, max=MAX_DIMENSION_LITERAL,
                            help="Output width. Omit it to infer the width from what follows."),
        "in_features": Arg(int, inferable=True, dim="D_in", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Input width. Normally inferred from the incoming tensor."),
        "bias": Arg(bool, True, positional=False, help="Add a learned bias vector."),
    },
    examples=[
        Example("linear(64)\nrelu()\nlinear()", ("B", 128), ("B", 10),
                "The final width is inferred from the output contract."),
        Example("linear(32, bias=False)", ("B", 16), ("B", 32)),
    ],
    category="core",
)
class Linear(nn.Linear):
    """Computes ``out = x @ weight.T + bias`` with ``weight`` of shape
    ``[out_features, in_features]``. No activation is applied; add one
    explicitly. Parameters are ``weight`` and, when enabled, ``bias``.
    """

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
