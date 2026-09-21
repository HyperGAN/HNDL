from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, INTS, operator


def _validate(args):
    dims = args["dims"]
    if not dims:
        raise HNDLError("E_ARGUMENT", "permute needs the new order of the non-batch axes, for example permute(2, 1)")
    if len(dims) > 3:
        raise HNDLError("E_ARGUMENT", "permute lists at most three axes; the deepest supported rank is 4")
    if sorted(dims) != list(range(1, len(dims) + 1)):
        raise HNDLError("E_ARGUMENT", f"permute dims {tuple(dims)} must be a permutation of "
                                      f"{tuple(range(1, len(dims) + 1))}; the batch axis 0 stays first")


def _relation(s):
    dims = s.args["dims"]
    rank = len(dims) + 1
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) != rank:
        s.error("E_ARGUMENT", f"permute lists {len(dims)} axes, which needs rank {rank}, but the tensor has "
                              f"rank {len(shape)}")
    s.rank("x", rank)
    s.rank("out", rank)
    for axis, source in enumerate(dims, start=1):
        s.axis("out", axis, s.shape("x")[source])
        s.axis("x", source, s.shape("out")[axis])


def _reference(module):
    sources, destinations = list(module.dims), list(range(1, len(module.dims) + 1))
    return lambda x: x.movedim(sources, destinations)


@operator(
    "permute",
    summary="Reorder the non-batch axes, keeping the batch axis first.",
    shape="x -> out",
    relation=_relation,
    reference=_reference,
    shape_text="out[i] == x[dims[i - 1]] for every non-batch axis i; rank == len(dims) + 1",
    args={
        "dims": Arg(INTS, (), min=1, max=3,
                    help="The non-batch axes 1..rank-1 of the input in their new order, given positionally: "
                         "permute(3, 1, 2) sends axis 3 to position 1. The batch axis 0 is implicit and "
                         "cannot be listed. The length fixes the rank."),
    },
    positional_rest="dims",
    validate=_validate,
    examples=[
        Example("permute(2, 1)", ("B", 16, 32), ("B", 32, 16),
                "A rank-3 swap: the [B, T, D] sequence becomes the [B, C, L] layout."),
        Example("permute(3, 1, 2)", ("B", 3, 8, 16), ("B", 16, 3, 8),
                "Width moves to the front of the non-batch axes; channels and height follow."),
        Example("conv(4, kernel_size=3, padding=1)\npermute(2, 3, 1)", ("B", 3, 8, 8), ("B", 8, 8, 4),
                "[B, C, H, W] to a channels-last [B, H, W, C] reading of the same data."),
        Example("linear()\npermute(2, 1)", ("B", 4, 6), ("B", 10, 4),
                "Bidirectional: the projection width 10 is inferred backward through the reorder."),
    ],
    category="shape",
)
class Permute(nn.Module):
    """Returns ``x.permute(0, *dims)``. ``dims`` lists the non-batch axes of
    the input in the order they take in the output, so ``out`` axis ``i``
    carries ``x`` axis ``dims[i - 1]``: ``permute(3, 1, 2)`` turns
    ``[B, C, H, W]`` into ``[B, W, C, H]``. The batch axis is never listed and
    always stays first.

    ``dims`` must be a permutation of ``1..rank-1``, which means its length
    alone fixes the rank of both ports: ``permute(2, 1)`` is rank 3 and
    ``permute(2, 3, 1)`` is rank 4. A list that is not a permutation of that
    range, one that includes the batch axis 0, or one whose length disagrees
    with a rank the neighbouring operations already fixed, is reported as
    ``E_ARGUMENT``. Rank 4 is ``[B, C, H, W]``; rank 3 is ``[B, T, D]`` for
    the sequence operations and ``[B, C, L]`` for 1-D convolution, and
    ``permute(2, 1)`` bridges those two readings.

    The result is a view sharing storage and autograd history with the input,
    generally non-contiguous; ``reshape`` and ``flatten`` copy when they must.
    There are no parameters and no train/eval difference, and the stride
    permutation is exact in every compute dtype.
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims)
        self.order = (0, *self.dims)

    def forward(self, x):
        return x.permute(self.order)

    def extra_repr(self):
        return f"dims={self.dims}"
