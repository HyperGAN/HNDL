from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) == 4:
        s.error("E_CONSTRAINT", "linear expects [B, D] or [B, T, D]; flatten or reshape image tensors first")


@operator(
    "linear",
    summary="Fully connected layer: a learned affine map on the last axis.",
    shape="x[B, ..., D_in] -> out[B, ..., D_out]",
    relation=_relation,
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
        Example("linear(64)\nrelu()\nlinear()", ("B", 16, 32), ("B", 16, 8),
                "On a [B, T, D] sequence the map applies to every position."),
    ],
    category="core",
)
class Linear(nn.Linear):
    """Computes ``out = x @ weight.T + bias`` with ``weight`` of shape
    ``[out_features, in_features]``, applied to the last axis of ``[B, D]``
    or ``[B, T, D]`` inputs. No activation is applied; add one explicitly.
    Parameters are ``weight`` and, when enabled, ``bias``.
    """

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
