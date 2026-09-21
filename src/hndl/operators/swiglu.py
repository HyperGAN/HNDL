from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) == 4:
        s.error("E_CONSTRAINT", "swiglu expects [B, D] or [B, T, D]; flatten or reshape image tensors first")


def _reference(module):
    gate, up, down = module.gate, module.up, module.down

    def run(x):
        activated = F.silu(F.linear(x, gate.weight, gate.bias))
        return F.linear(activated * F.linear(x, up.weight, up.bias), down.weight, down.bias)

    return run


@operator(
    "swiglu",
    summary="SwiGLU feed-forward block: a SiLU-gated projection folded back to the input width.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    relation=_relation,
    reference=_reference,
    args={
        "hidden": Arg(int, min=1, max=MAX_DIMENSION_LITERAL,
                      help="Width of the gated inner layer; both inner projections use it."),
        "bias": Arg(bool, False, positional=False,
                    help="Add a learned bias to all three projections. Gated blocks usually omit it."),
    },
    examples=[
        Example("swiglu(64)", ("B", 32), ("B", 32),
                "The block preserves the feature width, so it drops into any position."),
        Example("linear(32)\nswiglu(96)\nlinear()", ("B", 16), ("B", 10),
                "The inner width is explicit; the surrounding widths are inferred."),
        Example("swiglu(48, bias=True)", ("B", 8, 24), ("B", 8, 24),
                "On a [B, T, D] sequence the block acts on the last axis, independently per position."),
    ],
    category="sequence",
)
class SwiGLU(nn.Module):
    """The gated feed-forward block used by PaLM- and LLaMA-style transformers.
    Two projections read the same input: one is passed through SiLU and gates
    the other elementwise, and a third projection folds the product back to the
    input width.

    ```text
    out = down(silu(gate(x)) * up(x))
    silu(v) = v * sigmoid(v)
    ```

    The last axis carries the features `D`; every leading axis is a batch or
    position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
    applied independently at each position. Image tensors `[B, C, H, W]` are
    rejected with `E_CONSTRAINT`; flatten or reshape them first.

    Submodules are `gate` and `up` (both `Linear(D, hidden)`) and `down`
    (`Linear(hidden, D)`). Without biases the block holds `3 * D * hidden`
    parameters; with `bias=True` it holds `3 * D * hidden + 2 * hidden + D`.
    Because a gated block spends three matrices where a plain `feed_forward`
    spends two, `hidden` is commonly set to about two thirds of the width a
    plain block would use, for equal parameter count.

    There is no dropout and no normalization here, and no running state, so
    train and eval mode behave identically. Everything is computed in the
    incoming dtype, with no upcasting.
    """

    def __init__(self, hidden, bias, *, D):
        super().__init__()
        self.gate = nn.Linear(D, hidden, bias=bias)
        self.up = nn.Linear(D, hidden, bias=bias)
        self.down = nn.Linear(hidden, D, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
