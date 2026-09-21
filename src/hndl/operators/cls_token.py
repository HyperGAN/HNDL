import torch
from torch import nn

from ..operator import Example, operator


def _relation(s):
    x, out = s.shape("x"), s.shape("out")
    # The width check runs first so a contradictory contract reports why.
    if out is not None and out[1] is not None:
        if out[1] < 2:
            s.error("E_CONSTRAINT", f"cls_token emits at least two positions; the contract requires {out[1]}")
        s.axis("x", 1, out[1] - 1)
    if x is not None and x[1] is not None:
        s.axis("out", 1, x[1] + 1)


@operator(
    "cls_token",
    summary="Prepend one learned classification token to a sequence.",
    shape="x[B, T, D] -> out[B, T_plus, D]",
    relation=_relation,
    shape_text="T_plus == T + 1; the feature width D is unchanged",
    examples=[
        Example("cls_token()\nlinear(8)", ("B", 4, 6), ("B", 5, 8),
                "Four positions become five; the projection follows the token."),
        Example("embedding(64, 8)\ncls_token()\nlinear(4)", ("B", 6), ("B", 7, 4),
                "A token sequence gains a summary position before the projection.",
                input_dtype="int64"),
        Example("linear(12)\ncls_token()\npos_embed(16)\nlinear()", ("B", 4, 6), ("B", 5, 3),
                "The class token is counted by the position table that follows it."),
    ],
    category="sequence",
)
class ClsToken(nn.Module):
    """Prepends one learned vector to the position axis of a sequence:

    ```text
    out = cat([token.expand(B, 1, D), x], dim=1)
    out[:, 0, :] = token        # the class position
    out[:, 1:, :] = x           # the original sequence
    ```

    ``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features
    and the result has ``T + 1`` positions, so every downstream operator that
    counts positions — a position table, a pooling step, an output contract —
    sees the extra slot. The relation is bidirectional: a known input length
    fixes the output length and a known output length fixes the input length.

    The single parameter is ``token`` of shape ``[1, 1, D]``, initialized from
    a normal distribution with standard deviation 0.02 and broadcast across
    the batch, so every example starts from the same learned summary vector.
    It is a plain concatenation with no projection and no normalization;
    train and eval behave identically.
    """

    def __init__(self, *, D):
        super().__init__()
        self.dim = int(D)
        self.token = nn.Parameter(torch.empty(1, 1, self.dim))
        nn.init.normal_(self.token, mean=0.0, std=0.02)

    def forward(self, x):
        return torch.cat((self.token.expand(x.shape[0], 1, self.dim), x), dim=1)

    def extra_repr(self):
        return f"dim={self.dim}"
