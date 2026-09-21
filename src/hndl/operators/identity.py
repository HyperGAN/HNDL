from torch import nn

from ..operator import Example, operator


def _reference(module):
    def identity(x):
        return x

    return identity


@operator(
    "identity",
    summary="Pass the tensor through unchanged, as a named node.",
    shape="x[B, ...] -> out[B, ...]",
    examples=[
        Example('h = identity(name="hidden")\nlinear(8)\nrelu()\ny = linear(4)\nadd(y, h)', ("B", 4), ("B", 4),
                "A named tap: the passthrough gives a residual add something to refer back to."),
        Example("linear(64)\nidentity()\nrelu()\nlinear()", ("B", 32), ("B", 10),
                "A no-op placeholder that keeps the shape contract flowing through the chain."),
        Example("conv(8, kernel_size=3, padding=1)\nidentity()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Any supported rank passes through unchanged."),
    ],
    category="activation",
    reference=_reference,
)
class Identity(nn.Identity):
    """Returns `x` unchanged: `out = x`, the same tensor object, with no copy,
    no parameters, and no effect on the autograd graph beyond passing the
    gradient straight through.

    It exists so a configuration can name a point in the graph — `h =
    identity(name="hidden")` gives a branch or a later `add` something to refer
    to — and so a slot in a chain can be filled without changing the
    computation, for instance when a normalization or activation is being
    ablated. Shape, dtype, and device are preserved exactly for any supported
    rank: rank 2 `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`.
    Because the shape relation is the identity, it is fully transparent to
    resolution in both directions. Behavior is identical in train and eval
    mode.
    """

    def __init__(self):
        super().__init__()
