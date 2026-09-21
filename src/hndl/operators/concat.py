import torch
from torch import nn

from ..operator import Arg, Example, operator


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
    for dim in range(1, len(known)):
        if dim != axis:
            # Every tensor shares this extent. Propagate one known value once
            # per edge; comparing every pair would be quadratic work.
            extent = next((s.shape(port)[dim] for port in all_ports if s.shape(port)[dim] is not None), None)
            for port in all_ports:
                s.axis(port, dim, extent)
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
    shape_text="out[axis] == sum(x_i[axis]); all other axes equal",
    args={
        "input_count": Arg(int, min=2, positional=False, help="Number of tensors joined; set from the call."),
        "axis": Arg(int, 1, min=1, positional=False, help="Axis to concatenate; the batch axis 0 is excluded."),
    },
    examples=[
        Example("a, b = split(2)\nconcat(b, a)", ("B", 5), ("B", 5), "Reorder sections by concatenating them back."),
        Example("a = linear(4)\nb = linear(x)\nconcat(a, b)", ("B", 8), ("B", 10), "The second width 6 is inferred."),
    ],
    category="join",
)
class Concat(nn.Module):
    """``torch.cat(inputs, dim=axis)``. Every input is explicit; one missing
    extent along ``axis`` can be inferred from the output contract."""

    def __init__(self, input_count, axis):
        super().__init__()
        self.input_count = input_count
        self.axis = axis

    def forward(self, *xs):
        return torch.cat(xs, dim=self.axis)

    def extra_repr(self):
        return f"input_count={self.input_count}, axis={self.axis}"
