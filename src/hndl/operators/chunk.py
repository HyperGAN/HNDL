import torch
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_PORTS, operator
from ..types import MAX_BATCH_MULTIPLE, batch_dimension, batch_units


def _split_batch(s, chunks):
    """Along axis 0 every section takes an equal share of the batch.

    Batch entries count plan batches, so ``2*B`` divided into two sections is
    ``B`` each. A batch that cannot be divided --- one plan batch into two
    sections, or ``2*B`` into three --- is rejected here rather than deferred to
    a runtime that would only see one particular batch size.
    """
    incoming = s.batch("x")
    entries = [s.batch(port) for port in s.outputs]
    reference = next((entry for entry in (incoming, *entries) if entry is not None), None)
    if reference is None:
        return
    symbolic = type(reference) is str

    def entry(units):
        value = batch_dimension(units, symbolic)
        if value is None:
            s.error("E_RESOURCE", f"A batch axis of more than {MAX_BATCH_MULTIPLE} plan batches is not supported")
        return value

    if incoming is not None:
        units, remainder = divmod(batch_units(incoming), chunks)
        if remainder or units < 1:
            s.error("E_CONSTRAINT",
                    f"chunk cannot divide the batch {incoming!r} into {chunks} equal sections; a batch-axis chunk "
                    f"needs an input batch that is a multiple of {chunks}, such as the {chunks}*B that "
                    "concat(..., axis=0) produces")
    else:
        section = next((value for value in entries if value is not None), None)
        if section is None:
            return
        units = batch_units(section)
    for port in s.outputs:
        s.axis(port, 0, entry(units))
    s.axis("x", 0, entry(units * chunks))


def _relation(s):
    ports = ("x", *s.outputs)
    chunks, dim = s.args["chunks"], s.args["dim"]
    known = next((s.shape(port) for port in ports if s.shape(port) is not None), None)
    if known is None:
        return
    rank = len(known)
    if dim >= rank:
        s.error("E_ARGUMENT", f"chunk dim {dim} is outside rank {rank}")
    for port in ports:
        s.rank(port, rank)
    if dim == 0:
        _split_batch(s, chunks)
    else:
        s.share_batch(*ports)
    for axis in range(1, rank):
        if axis == dim:
            continue
        extent = next((s.shape(port)[axis] for port in ports if s.shape(port)[axis] is not None), None)
        for port in ports:
            s.axis(port, axis, extent)
    if dim == 0:
        return
    section = next((s.shape(port)[dim] for port in s.outputs if s.shape(port)[dim] is not None), None)
    extent = s.shape("x")[dim]
    if extent is not None:
        size, remainder = divmod(extent, chunks)
        if remainder or size < 1:
            s.error("E_CONSTRAINT", f"chunk cannot divide extent {extent} on dim {dim} into {chunks} equal sections")
        section = size
    if section is None:
        return
    for port in s.outputs:
        s.axis(port, dim, section)
    s.axis("x", dim, section * chunks)


@operator(
    "chunk",
    summary="Cut one axis into a fixed number of equal sections.",
    shape="x -> out*",
    outputs_from="chunks",
    relation=_relation,
    batch="relation",
    shape_text="out_i[dim] == x[dim] / chunks for every i; all other axes equal",
    args={
        "chunks": Arg(int, min=1, max=MAX_PORTS,
                      help="Number of equal sections, which is also the number of outputs."),
        "dim": Arg(int, 1, min=0, positional=False,
                   help="Axis to cut; 0 divides the batch, which must be a multiple of chunks."),
    },
    examples=[
        Example("a, b = chunk(2)\nconcat(b, a)", ("B", 8), ("B", 8),
                "Two equal halves of the feature axis, concatenated back in the other order."),
        Example("pair = concat(x, x, axis=0)\nshared = linear(pair, 6)\np, q = chunk(shared, 2, dim=0)\nconcat(p, q)",
                ("B", 4), ("B", 12),
                "One linear runs over both copies at batch 2*B; chunk along dim 0 returns the pair at batch B."),
        Example("a, b, c = chunk(3, dim=2)\nconcat(a, c, b, axis=2)", ("B", 4, 9), ("B", 4, 9),
                "Three equal sections of a rank-3 tensor, reordered."),
    ],
    category="shape",
)
class Chunk(nn.Module):
    """Returns ``chunks`` equal sections of ``dim`` as views sharing autograd
    with the input, in order, from ``torch.split`` with one exact size.

    ``chunk`` is the equal-split counterpart of ``split``, which cuts one
    tensor into a first section and the remainder. The axis must divide
    exactly: an extent of 9 into 2 sections fails with ``E_CONSTRAINT`` at
    resolution, not at runtime, and no section may be empty.

    ``dim=0`` divides the batch. Batch is symbolic, so the division is done on
    batch entries: a ``2*B`` input --- what ``concat(..., axis=0)`` of two
    ``B`` tensors produces --- chunks into two ``B`` outputs, and ``3*B`` into
    three. A plain ``B`` input cannot be chunked, because nothing says the
    runtime batch is even; that fails at resolution with ``E_CONSTRAINT``. The
    module still checks divisibility in ``forward`` (``E_RUNTIME``) for a
    submodule called directly, outside the graph that checked its contract::

        combined = concat(candidate, context, axis=0)   # [2*B, C, H, W]
        features = pretrained(combined, ...)            # one shared pass
        a, b = chunk(features, 2, dim=0)                # [B, ...] and [B, ...]

    Like every multi-output call, ``chunk`` returns a tuple and clears the
    current tensor, so the next operation must name its input. ``chunks=1`` is
    allowed and returns a one-tuple containing the whole tensor, which also
    clears current. Both first and second derivatives flow through every
    section: the sections are views, not copies.
    """

    def __init__(self, chunks, dim, *, input_shapes):
        super().__init__()
        self.chunks = chunks
        self.dim = dim
        # Non-batch extents are fixed by the plan; the batch axis is not, so
        # its section size is computed per call from the tensor itself.
        self.size = None if dim == 0 else input_shapes["x"][dim] // chunks

    def forward(self, x):
        size, remainder = divmod(x.shape[self.dim], self.chunks)
        if remainder or size < 1:
            raise HNDLError("E_RUNTIME", f"chunk: extent {x.shape[self.dim]} on dim {self.dim} does not divide "
                                         f"into {self.chunks} equal sections")
        return torch.split(x, size, dim=self.dim)

    def extra_repr(self):
        return f"chunks={self.chunks}, dim={self.dim}"
