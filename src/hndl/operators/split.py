import torch
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    ports = ("x", "first", "rest")
    known = next((s.shape(port) for port in ports if s.shape(port) is not None), None)
    if known is None:
        return
    dim = s.args["dim"]
    if dim >= len(known):
        s.error("E_ARGUMENT", f"split dim {dim} is outside rank {len(known)}")
    for port in ports:
        s.rank(port, len(known))
    for axis in range(1, len(known)):
        if axis != dim:
            for target in ports:
                for source in ports:
                    s.axis(target, axis, s.shape(source)[axis])
    s.axis("first", dim, s.args.get("size"))
    i, a, b = (s.shape(port)[dim] for port in ports)
    if a is not None and b is not None:
        s.axis("x", dim, a + b)
    if i is not None and a is not None:
        s.axis("rest", dim, i - a)
    if i is not None and b is not None:
        s.axis("first", dim, i - b)
    s.arg("size", s.shape("first")[dim])


@operator(
    "split",
    summary="Cut one axis into a first section and the remainder.",
    shape="x -> first, rest",
    relation=_relation,
    shape_text="first[dim] == size; rest[dim] == x[dim] - size; all other axes equal",
    args={
        "size": Arg(int, inferable=True, min=1, max=MAX_DIMENSION_LITERAL,
                    help="Extent of the first section. Inferred when the consumers determine it."),
        "dim": Arg(int, 1, min=1, positional=False, help="Axis to split; the batch axis 0 cannot be split."),
    },
    examples=[
        Example("z1, z2 = split(64)\nlinear(z1, 32)\nfeatures = relu()\nstyle = linear(z2, 32)\nadd(features, style)",
                ("B", 128), ("B", 32), "Both sections feed explicit branches; split clears the current tensor."),
        Example("a, b = split()\nout = b", ("B", 128), ("B", 32), "The first size 96 is inferred from the remainder."),
    ],
    category="shape",
)
class Split(nn.Module):
    """Returns ``(x[:size], x[size:])`` along ``dim`` as views sharing autograd
    with the input. Exactly two non-empty sections are produced; this is not
    a repeated chunking. After ``split`` there is no single current tensor, so
    the next operation must name its input.
    """

    def __init__(self, size, dim, *, input_shapes):
        super().__init__()
        self.sizes = (size, input_shapes["x"][dim] - size)
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.sizes, dim=self.dim)

    def extra_repr(self):
        return f"sizes={self.sizes}, dim={self.dim}"
