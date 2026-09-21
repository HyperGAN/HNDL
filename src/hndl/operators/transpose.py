from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    dim0, dim1 = s.args["dim0"], s.args["dim1"]
    highest = max(dim0, dim1)
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        # Swapping axis 3 is only possible at rank 4, so the rank is known
        # from the arguments alone even before a neighbour supplies a shape.
        if highest == 3:
            s.rank("x", 4)
            s.rank("out", 4)
        return
    rank = len(shape)
    if highest >= rank:
        s.error("E_ARGUMENT", f"transpose dim {highest} is outside rank {rank}")
    s.rank("x", rank)
    s.rank("out", rank)
    swap = {dim0: dim1, dim1: dim0}
    for axis in range(1, rank):
        source = swap.get(axis, axis)
        s.axis("out", axis, s.shape("x")[source])
        s.axis("x", source, s.shape("out")[axis])


def _reference(module):
    dim0, dim1 = module.dim0, module.dim1
    if dim0 == dim1:
        return lambda x: x
    return lambda x: x.movedim((dim0, dim1), (dim1, dim0))


@operator(
    "transpose",
    summary="Swap two non-batch axes of the tensor.",
    shape="x -> out",
    relation=_relation,
    reference=_reference,
    shape_text="out[dim0] == x[dim1]; out[dim1] == x[dim0]; every other axis is unchanged",
    args={
        "dim0": Arg(int, min=1, max=3, help="First axis of the swap; 1 is the first non-batch axis and 3 is the "
                                            "last axis of a rank-4 tensor. The batch axis 0 cannot be swapped."),
        "dim1": Arg(int, min=1, max=3, help="Second axis of the swap. Equal to dim0 means the identity."),
    },
    examples=[
        Example("transpose(1, 2)", ("B", 16, 32), ("B", 32, 16),
                "Bridges a [B, T, D] sequence to the [B, C, L] layout 1-D convolutions expect."),
        Example("transpose(2, 3)", ("B", 3, 8, 16), ("B", 3, 16, 8),
                "Swaps height and width of an image tensor, leaving the channel axis alone."),
        Example("linear()\ntranspose(1, 2)", ("B", 4, 6), ("B", 10, 4),
                "The swap is bidirectional: the projection width 10 is read back through it."),
    ],
    category="shape",
)
class Transpose(nn.Module):
    """Returns ``x.transpose(dim0, dim1)``: the extents at ``dim0`` and
    ``dim1`` trade places and every other axis keeps its extent. The batch
    axis 0 is never part of a swap, so both arguments are at least 1 and at
    most 3 (the last axis of the deepest supported rank). Passing the same
    axis twice is the identity.

    Axis conventions. A rank-3 tensor is ``[B, T, D]`` for the sequence
    operations — ``linear`` and the activations act on the last axis, ``D``,
    across ``T`` positions — while 1-D convolution and pooling read rank 3 as
    ``[B, C, L]``, channels before length. ``transpose(1, 2)`` is the bridge
    between the two readings, and a second ``transpose(1, 2)`` afterwards
    returns to the sequence layout. At rank 4 the layout is ``[B, C, H, W]``,
    so ``transpose(2, 3)`` swaps height and width and ``transpose(1, 3)``
    exchanges channels with width.

    The result is a view: it shares storage and autograd history with the
    input and is generally not contiguous. Downstream operations handle
    non-contiguous inputs; ``reshape`` and ``flatten`` copy when they must.
    There are no parameters, no buffers, and no difference between train and
    eval mode. The computation is a stride permutation, so it is exact in
    every compute dtype.
    """

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self):
        return f"dim0={self.dim0}, dim1={self.dim1}"
