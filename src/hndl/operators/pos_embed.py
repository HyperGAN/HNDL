import torch
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        return
    positions = shape[1]
    max_len = s.args.get("max_len")
    if positions is not None and max_len is not None and positions > max_len:
        s.error("E_CONSTRAINT",
                f"pos_embed(max_len={max_len}) cannot address {positions} positions; raise max_len")


@operator(
    "pos_embed",
    summary="Add a learned position vector to every position of a sequence.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    shape_text="out == x elementwise; T <= max_len",
    args={
        "max_len": Arg(int, min=1, max=MAX_DIMENSION_LITERAL,
                       help="Rows in the position table; the longest sequence this layer can encode."),
    },
    examples=[
        Example("pos_embed(64)\nlinear(8)", ("B", 6, 4), ("B", 6, 8),
                "Positions 0..5 of the table are added; rows 6..63 stay unused."),
        Example("embedding(500, 16)\npos_embed(32)\nlinear(4)", ("B", 12), ("B", 12, 4),
                "The usual transformer input stem: token embeddings plus learned positions.",
                input_dtype="int64"),
    ],
    category="sequence",
)
class PosEmbed(nn.Module):
    """A learned absolute position encoding, added to the sequence:

    ```text
    out[b, t, :] = x[b, t, :] + weight[t, :]
    ```

    ``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features;
    the same position vector is added to every example in the batch. The
    parameter ``weight`` has shape ``[max_len, D]`` and is initialized from a
    normal distribution with standard deviation 0.02, the convention for
    transformer position tables. Only the first ``T`` rows take part in the
    forward pass, so only those rows receive a gradient; the remaining rows
    are kept so that the same module can encode longer sequences after a
    re-resolve.

    ``max_len`` is a capacity, not a shape: the resolver only requires
    ``T <= max_len`` and reports ``E_CONSTRAINT`` when a resolved sequence is
    longer than the table. Train and eval behave identically, and the addition
    is performed in the input dtype.
    """

    def __init__(self, max_len, *, D):
        super().__init__()
        self.max_len = int(max_len)
        self.dim = int(D)
        self.weight = nn.Parameter(torch.empty(self.max_len, self.dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.weight[:x.shape[1]]

    def extra_repr(self):
        return f"max_len={self.max_len}, dim={self.dim}"
