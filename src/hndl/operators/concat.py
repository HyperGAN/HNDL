import torch
from torch import nn

from ..operator import Arg, Example, operator
from ..types import MAX_BATCH_MULTIPLE, batch_dimension, batch_units


def _join_batch(s, refs):
    """Along axis 0 the output batch is the sum of the inputs' batches.

    Batch entries count plan batches: ``"B"`` is one and ``"2*B"`` is two, so
    ``B + B`` is ``2*B`` and ``2*B + B`` is ``3*B``. One unknown input batch is
    solved from a known output, exactly as one unknown extent is on any other
    axis.
    """
    entries = [s.batch(port) for port in refs]
    total_entry = s.batch("out")
    reference = next((entry for entry in (*entries, total_entry) if entry is not None), None)
    if reference is None:
        return
    symbolic = type(reference) is str

    def entry(units):
        value = batch_dimension(units, symbolic)
        if value is None:
            s.error("E_RESOURCE", f"A batch-axis join of more than {MAX_BATCH_MULTIPLE} plan batches "
                                  "is not supported")
        return value

    units = [None if value is None else batch_units(value) for value in entries]
    missing = [index for index, value in enumerate(units) if value is None]
    if not missing:
        s.axis("out", 0, entry(sum(units)))
    elif len(missing) == 1 and total_entry is not None:
        remaining = batch_units(total_entry) - sum(value for value in units if value is not None)
        if remaining < 1:
            s.error("E_CONSTRAINT", f"Joined batch {total_entry!r} leaves nothing for {refs[missing[0]]}")
        s.axis(refs[missing[0]], 0, entry(remaining))


def _relation(s):
    refs = tuple(s.inputs)
    all_ports = refs + ("out",)
    known = next((s.shape(port) for port in all_ports if s.shape(port) is not None), None)
    if known is None:
        return
    axis = s.args["axis"]
    if axis >= len(known):
        s.error("E_ARGUMENT", f"concat axis {axis} is outside rank {len(known)}")
    for port in all_ports:
        s.rank(port, len(known))
    if axis == 0:
        _join_batch(s, refs)
    else:
        s.share_batch(*all_ports)
    for dim in range(1, len(known)):
        if dim != axis:
            # Every tensor shares this extent. Propagate one known value once
            # per edge; comparing every pair would be quadratic work.
            extent = next((s.shape(port)[dim] for port in all_ports if s.shape(port)[dim] is not None), None)
            for port in all_ports:
                s.axis(port, dim, extent)
    if axis == 0:
        return
    sizes = [s.shape(port)[axis] for port in refs]
    missing = [i for i, size in enumerate(sizes) if size is None]
    if not missing:
        s.axis("out", axis, sum(sizes))
    elif len(missing) == 1 and s.shape("out")[axis] is not None:
        s.axis(refs[missing[0]], axis, s.shape("out")[axis] - sum(size for size in sizes if size is not None))


@operator(
    "concat",
    summary="Join two or more tensors along one axis.",
    shape="x* -> out",
    relation=_relation,
    batch="relation",
    shape_text="out[axis] == sum(x_i[axis]); all other axes equal",
    args={
        "input_count": Arg(int, min=2, positional=False, help="Number of tensors joined; set from the call."),
        "axis": Arg(int, 1, min=0, positional=False,
                    help="Axis to concatenate; axis 0 joins along batch and adds the inputs' batches."),
    },
    examples=[
        Example("a, b = split(2)\nconcat(b, a)", ("B", 5), ("B", 5), "Reorder sections by concatenating them back."),
        Example("a = linear(4)\nb = linear(x)\nconcat(a, b)", ("B", 8), ("B", 10), "The second width 6 is inferred."),
        Example("a = linear(4)\nb = linear(x, 4)\npair = concat(a, b, axis=0)\nshared = linear(pair, 6)\n"
                "p, q = chunk(shared, 2, dim=0)\nconcat(p, q)",
                ("B", 8), ("B", 12),
                "One shared linear runs over both branches at batch 2*B; chunk takes the pair apart."),
    ],
    category="shape",
)
class Concat(nn.Module):
    """``torch.cat(inputs, dim=axis)``. Every input is explicit; one missing
    extent along ``axis`` can be inferred from the output contract.

    ``axis=0`` joins along batch: the inputs must agree on every other axis and
    the result carries their batches added together, written ``2*B`` for two
    equal batches. That is how one shared --- often frozen --- network runs over
    two branches in a single forward pass; ``chunk(..., dim=0)`` takes the
    result apart again.
    """

    def __init__(self, input_count, axis):
        super().__init__()
        self.input_count = input_count
        self.axis = axis

    def forward(self, *xs):
        return torch.cat(xs, dim=self.axis)

    def extra_repr(self):
        return f"input_count={self.input_count}, axis={self.axis}"
