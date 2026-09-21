from torch import nn

from ..operator import Example, operator


def _relation(s):
    s.product("x", "out")


@operator(
    "flatten",
    summary="Collapse every non-batch axis into one feature axis.",
    shape="x[B, ...] -> out[B, F]",
    relation=_relation,
    shape_text="F == prod(x[1:])",
    examples=[Example("flatten()\nlinear()", ("B", 3, 8, 8), ("B", 2))],
    category="shape",
)
class Flatten(nn.Flatten):
    """Equivalent to ``x.reshape(batch, -1)``. The input rank can be inferred
    backward only through the element count when the other axes are known."""

    def __init__(self):
        super().__init__(start_dim=1)
