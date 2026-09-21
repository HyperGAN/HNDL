from torch import nn

from ..operator import Example, operator


@operator(
    "add",
    summary="Elementwise sum of two tensors with identical shapes.",
    shape="a[B, ...], b[B, ...] -> out[B, ...]",
    examples=[Example('saved = x\nlinear(8, name="branch")\nrelu()\nh = linear(4)\nadd(h, saved)',
                      ("B", 4), ("B", 4), "A residual connection: both inputs are explicit.")],
    category="join",
)
class Add(nn.Module):
    """``out = a + b`` with no broadcasting; shapes, dtype, and batch must
    match exactly. Both tensor inputs must be supplied explicitly."""

    def forward(self, a, b):
        return a + b
