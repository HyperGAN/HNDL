from torch import nn

from ..operator import Arg, Example, operator

MODES = ("mean", "first", "last", "max")


def _pool(mode, x):
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "first":
        return x[:, 0]
    if mode == "last":
        return x[:, -1]
    return x.max(dim=1).values


def _reference(module):
    mode = module.mode
    if mode == "mean":
        return lambda x: x.sum(dim=1) / x.shape[1]
    if mode == "first":
        return lambda x: x.select(1, 0)
    if mode == "last":
        return lambda x: x.select(1, x.shape[1] - 1)
    return lambda x: x.amax(dim=1)


@operator(
    "pool_tokens",
    summary="Reduce a [B, T, D] sequence to one [B, D] vector per example.",
    shape="x[B, T, D] -> out[B, D]",
    args={
        "mode": Arg(str, "mean", choices=MODES,
                    help="Reduction over the token axis: mean, first, last, or max."),
    },
    reference=_reference,
    examples=[
        Example("linear(16)\npool_tokens()\nlinear()", ("B", 6, 8), ("B", 10),
                "Averages the 6 token vectors, then classifies the pooled feature."),
        Example('pool_tokens("last")\nlinear()', ("B", 4, 32), ("B", 8),
                "Takes the final token, the usual pooling for a causal sequence model."),
        Example('linear(12)\nrelu()\npool_tokens(mode="max")\nlinear()', ("B", 5, 6), ("B", 3),
                "Elementwise maximum over the tokens."),
        Example('reshape(4)\nlinear(8)\npool_tokens("first")', ("B", 64), ("B", 8),
                "A leading [CLS]-style token is selected after the sequence is projected."),
    ],
    category="sequence",
)
class PoolTokens(nn.Module):
    """Collapses the token axis of a rank-3 sequence ``[B, T, D]`` into a
    single vector per example, giving ``[B, D]``. ``T`` is the position axis
    and ``D`` the feature axis; the feature axis is carried through unchanged,
    so the pooled width always equals the incoming width.

    ```text
    mean   out[b, d] = (1 / T) * sum_t x[b, t, d]
    first  out[b, d] = x[b, 0, d]
    last   out[b, d] = x[b, T - 1, d]
    max    out[b, d] = max_t x[b, t, d]
    ```

    ``mean`` averages every position, which is the default and the usual
    choice for bidirectional encoders. ``first`` picks the leading token, the
    ``[CLS]`` convention. ``last`` picks the final token, the convention for
    causal models whose sequences all end at the same position; it is the
    wrong choice when the useful position varies per example. ``max`` takes
    the elementwise maximum, resolving ties to the same value rather than to
    a position.

    There are no parameters and no state, so train and eval behave
    identically. The reduction runs in the incoming dtype: with ``mean`` in
    float16 a long sequence accumulates rounding error in the sum, which is
    inherent to computing in the plan's compute dtype.
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return _pool(self.mode, x)

    def extra_repr(self):
        return f"mode={self.mode!r}"
