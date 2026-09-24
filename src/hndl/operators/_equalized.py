"""Shared equalized-learning-rate affine projection (unit gain and LR multiplier)."""
from torch import nn
from torch.nn import functional as F


class EqualLinear(nn.Linear):
    """Store raw N(0,1) weights; apply 1/sqrt(fan_in) on every forward.

    Bias starts at zero and is not scaled. No fused activation is applied.
    ``equalized=False`` retains nn.Linear initialization and forward behavior.
    Construction overrides and checkpoints contain raw, not effective, weights.
    """

    def __init__(self, in_features, out_features, bias=True, *, equalized=True):
        self.equalized = equalized
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        if not self.equalized:
            return super().reset_parameters()
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if not self.equalized:
            return super().forward(x)
        return F.linear(x, self.weight * (self.in_features ** -0.5), self.bias)
