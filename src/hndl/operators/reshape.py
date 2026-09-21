from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, INTS, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    prefix = s.args["shape"]
    if len(prefix) >= 2:
        s.rank("out", 4)
    shape = s.shape("out")
    if shape is not None:
        if len(prefix) >= len(shape):
            s.error("E_RESHAPE", "Reshape prefix exceeds the output rank")
        for axis, value in enumerate(prefix, 1):
            s.axis("out", axis, value, "E_RESHAPE")
    s.product("x", "out")


def _validate(args):
    if len(args["shape"]) > 3:
        raise HNDLError("E_ARGUMENT", "reshape shape must be a prefix of at most three dimensions")


def _finalize(args, input_shapes, output_shapes):
    return {**args, "shape": tuple(output_shapes["out"][1:])}


@operator(
    "reshape",
    summary="View the tensor with new non-batch dimensions, preserving the element count.",
    shape="x -> out",
    relation=_relation,
    shape_text="prod(x[1:]) == prod(out[1:]); out[1:len(shape)+1] == shape",
    args={"shape": Arg(INTS, (), min=1, max=MAX_DIMENSION_LITERAL,
                       help="Leading non-batch dimensions. Remaining dimensions are inferred.")},
    positional_rest="shape",
    validate=_validate,
    finalize=_finalize,
    examples=[
        Example("linear()\nreshape(8, 4, 4)", ("B", 16), ("B", 8, 4, 4),
                "The projection width 128 follows from the reshape target."),
        Example("reshape(512)\nconv(3, kernel_size=1)", ("B", 8192), ("B", 3, 4, 4),
                "Height and width are inferred from the element count and the output contract."),
        Example("reshape()\nlinear()", ("B", 2, 4, 4), ("B", 10),
                "A bare reshape flattens when the consumer fixes rank 2."),
    ],
    category="shape",
)
class Reshape(nn.Module):
    """Returns ``x.reshape(batch, *shape)``. The batch axis is never reshaped.
    Give the leading dimensions positionally, ``reshape(512, 4, 4)``, or as
    ``shape=(512, 4, 4)``; exactly one omitted factor can be solved from the
    element count. The plan records the full resolved shape.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def extra_repr(self):
        return f"shape={self.shape}"
