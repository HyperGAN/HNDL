import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator

EPS = 1e-5


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) == 4:
        s.error("E_CONSTRAINT", "hopfield expects [B, D] or [B, T, D]; flatten or reshape image tensors first")


def _reference(module):
    """A handwritten retrieval loop the operator harness compares against."""
    stored, beta, steps, normalize = module.stored, module.beta, module.steps, module.normalize

    def run(x):
        xi = x
        if normalize:
            mean = xi.mean(dim=-1, keepdim=True)
            variance = xi.var(dim=-1, unbiased=False, keepdim=True)
            xi = (xi - mean) * torch.rsqrt(variance + EPS)
        for _ in range(steps):
            scores = beta * torch.einsum("...d,pd->...p", xi, stored)
            xi = torch.einsum("...p,pd->...d", torch.softmax(scores, dim=-1), stored)
        return xi

    return run


@operator(
    "hopfield",
    summary="Retrieve learned patterns by iterating the modern Hopfield update on the last axis.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    relation=_relation,
    args={
        "patterns": Arg(int, min=1, max=MAX_DIMENSION_LITERAL,
                        help="Number of stored patterns to learn; the rows of the memory matrix."),
        "beta": Arg(float, 1.0, min=0, exclusive_min=True, positional=False,
                    help="Inverse temperature of the softmax; larger values retrieve a single pattern more sharply."),
        "steps": Arg(int, 1, min=1, max=1024, positional=False,
                     help="Number of retrieval updates applied in sequence; one update equals attention."),
        "normalize": Arg(bool, True, positional=False,
                         help="Layer-normalize the query over the last axis before the first update."),
    },
    reference=_reference,
    examples=[
        Example("linear(32)\nhopfield(16)\nlinear()", ("B", 8), ("B", 4),
                "Sixteen stored patterns of width 32; the memory sits between two projections."),
        Example("hopfield(8, beta=4.0, steps=3)", ("B", 12), ("B", 12),
                "A sharp memory iterated three times retrieves close to a single stored pattern."),
        Example("linear(24)\nhopfield(12, normalize=False)\nlinear()", ("B", 6, 10), ("B", 6, 5),
                "On a [B, T, D] sequence every position queries the memory independently."),
        Example("linear(16)\nhopfield(4, beta=0.5)\ntanh()\nlinear()", ("B", 20), ("B", 3),
                "A soft memory mixes all four patterns and behaves like a bottleneck."),
    ],
    category="memory",
)
class Hopfield(nn.Module):
    """A modern (continuous) Hopfield layer, following Ramsauer et al.,
    *Hopfield Networks is All You Need* (2020).

    The layer owns a learned memory matrix ``stored`` of shape
    ``[patterns, D]`` whose rows are the stored patterns ``X``. For a query
    ``xi`` of width ``D`` it minimizes the energy

    ```text
    E(xi) = -lse(beta, X xi) + 0.5 * xi.T @ xi + log(patterns)/beta + 0.5 * M**2
    ```

    where ``lse`` is the log-sum-exp over the stored patterns and ``M`` is the
    largest pattern norm. The concave-convex procedure gives the update that
    this layer applies, repeated ``steps`` times:

    ```text
    xi <- X.T @ softmax(beta * X @ xi)
    ```

    Written as a batch of rows, ``xi <- softmax(beta * xi @ stored.T) @
    stored``. A single update is exactly dot-product attention with the query
    ``xi``, the stored patterns as both keys and values, and ``beta`` in place
    of the ``1/sqrt(d)`` scale: this layer is attention whose keys and values
    are parameters rather than activations. Iterating the update is the
    associative-memory reading of the same equation, and the fixed points are
    the stored patterns (or, at small ``beta``, their metastable averages).

    Retrieval behavior follows ``beta``. With ``beta`` large and the patterns
    well separated, a query near a stored pattern makes one softmax weight
    dominate, so the output is that pattern and further steps do not move it.
    With ``beta`` small the softmax is flat and the output tends toward the
    mean of the stored patterns; intermediate values retrieve averages of the
    patterns a query is close to. Because the energy decreases monotonically
    the iteration converges, in practice within a handful of updates, so
    ``steps > 1`` mainly sharpens retrieval rather than changing its nature.

    Axis convention: the layer acts on the last axis of ``[B, D]`` or
    ``[B, T, D]`` inputs, so every position of a sequence queries the same
    memory independently. Image tensors are rejected; flatten or reshape
    first. The output has the shape of the input.

    When ``normalize`` is true the query is layer-normalized over the last
    axis, without a learned affine and with ``eps = 1e-5``, before the first
    update only; later updates consume the previous retrieval unchanged. The
    normalization keeps ``beta * X @ xi`` in a usable range when the incoming
    activations have drifted in scale.

    The only parameter is ``stored``, initialized from ``normal(0, 1/sqrt(D))``
    so that initial scores have unit scale. Behavior is identical in train and
    eval mode; there are no buffers and no running statistics. Everything is
    computed in the input dtype, except that the layer normalization follows
    ``torch.nn.functional.layer_norm``, which accumulates its statistics in
    float32 for float16 and bfloat16 inputs.
    """

    def __init__(self, patterns, beta, steps, normalize, *, D):
        super().__init__()
        self.patterns = int(patterns)
        self.features = int(D)
        self.beta = float(beta)
        self.steps = int(steps)
        self.normalize = bool(normalize)
        self.stored = nn.Parameter(torch.randn(self.patterns, self.features) * self.features**-0.5)

    def forward(self, x):
        xi = F.layer_norm(x, (self.features,), eps=EPS) if self.normalize else x
        for _ in range(self.steps):
            xi = torch.softmax(self.beta * (xi @ self.stored.transpose(0, 1)), dim=-1) @ self.stored
        return xi

    def extra_repr(self):
        return (f"patterns={self.patterns}, features={self.features}, beta={self.beta}, "
                f"steps={self.steps}, normalize={self.normalize}")
