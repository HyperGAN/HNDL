import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator
from ._equalized import EqualLinear

ACTIVATIONS = {
    "gelu": lambda h: F.gelu(h),
    "gelu_tanh": lambda h: F.gelu(h, approximate="tanh"),
    "relu": lambda h: F.relu(h),
    "silu": lambda h: F.silu(h),
    "quick_gelu": lambda h: h * torch.sigmoid(1.702 * h),
}


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) == 4:
        s.error("E_CONSTRAINT", "feed_forward expects [B, D] or [B, T, D]; flatten or reshape image tensors first")


def _reference(module):
    activation = ACTIVATIONS[module.activation]
    up, down, p = module.up, module.down, module.dropout.p

    def run(x):
        up_gain = up.in_features ** -0.5 if getattr(up, 'equalized', False) else 1
        down_gain = down.in_features ** -0.5 if getattr(down, 'equalized', False) else 1
        hidden = activation(F.linear(x, up.weight * up_gain, up.bias))
        if p:
            hidden = F.dropout(hidden, p, module.training)
        return F.linear(hidden, down.weight * down_gain, down.bias)

    return run


@operator(
    "feed_forward",
    summary="Transformer feed-forward block: widen, activate, project back.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    relation=_relation,
    reference=_reference,
    args={
        "hidden": Arg(int, min=1, max=MAX_DIMENSION_LITERAL,
                      help="Width of the inner layer; typically two to four times the feature width D."),
        "activation": Arg(str, "gelu", positional=False, choices=tuple(ACTIVATIONS),
                          help="Nonlinearity applied to the inner activations."),
        "dropout": Arg(float, 0.0, min=0, max=1, exclusive_max=True, positional=False,
                       help="Dropout probability applied after the activation; 0 disables it."),
        "bias": Arg(bool, True, positional=False, help="Add a learned bias to both projections."),
        "equalized": Arg(bool, False, positional=False,
                         help="Use runtime fan-in scaling and N(0,1) raw weights for both linear projections."),
    },
    examples=[
        Example("feed_forward(64)", ("B", 32), ("B", 32),
                "The block preserves the feature width, so it drops into any position."),
        Example("feed_forward(64, equalized=True)", ("B", 32), ("B", 32),
                "Equalized projections preserve the selected activation and dropout."),
        Example('linear(32)\nfeed_forward(96, activation="silu")\nlinear()', ("B", 16), ("B", 10),
                "The inner width is explicit; the surrounding widths are inferred."),
        Example('feed_forward(64, activation="gelu_tanh")', ("B", 8, 24), ("B", 8, 24),
                "On a [B, T, D] sequence the block acts on the last axis, independently per position."),
        Example("feed_forward(32, bias=False)", ("B", 4, 16), ("B", 4, 16),
                "Without biases the block holds exactly 2 * D * hidden parameters."),
    ],
    category="sequence",
)
class FeedForward(nn.Module):
    """The position-wise feed-forward network of a transformer block: a widening
    projection, a nonlinearity, optional dropout, and a projection back to the
    input width.

    ```text
    h   = activation(x @ up.weight.T + up.bias)      # [B, ..., hidden]
    out = dropout(h) @ down.weight.T + down.bias     # [B, ..., D]
    ```

    The last axis carries the features `D`; every leading axis is a batch or
    position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
    applied independently at each position. Image tensors `[B, C, H, W]` are
    rejected with `E_CONSTRAINT`; flatten or reshape them first.

    Submodules are `up` (`Linear(D, hidden)`), `dropout` and `down`
    (`Linear(hidden, D)`), so the parameters are `up.weight`, `up.bias`,
    `down.weight` and `down.bias`. With biases the block holds
    `2 * D * hidden + hidden + D` parameters, and `2 * D * hidden` without.

    ``equalized=True`` initializes both raw projection weights N(0,1), zeros
    their biases, and scales each weight by the inverse square root of its own
    fan-in (D for up, hidden for down) on every forward. Gain and learning-rate
    multiplier are one. Activations and dropout are unchanged. ``init=`` and
    checkpoints contain raw weights; parameter names and shapes stay the same.

    `activation` selects one of:

    | Name | Formula |
    | --- | --- |
    | `gelu` | `h * Phi(h)` with the exact Gaussian CDF |
    | `gelu_tanh` | the `tanh` approximation of GELU |
    | `relu` | `max(h, 0)` |
    | `silu` | `h * sigmoid(h)` |
    | `quick_gelu` | `h * sigmoid(1.702 * h)` |

    Dropout is active in train mode only; in eval mode, and whenever `dropout`
    is 0, the block is deterministic. Everything is computed in the incoming
    dtype, with no upcasting.
    """

    def __init__(self, hidden, activation, dropout, bias, *, D, equalized=False):
        super().__init__()
        self.activation = activation
        linear = EqualLinear if equalized else nn.Linear
        self.up = linear(D, hidden, bias=bias)
        self.dropout = nn.Dropout(float(dropout))
        self.down = linear(hidden, D, bias=bias)

    def forward(self, x):
        return self.down(self.dropout(ACTIVATIONS[self.activation](self.up(x))))

    def extra_repr(self):
        return f"activation={self.activation!r}"
