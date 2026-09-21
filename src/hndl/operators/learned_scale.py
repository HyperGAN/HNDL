import torch
from torch import nn

from ..operator import Arg, Example, operator


@operator(
    "learned_scale",
    summary="Multiply a tensor by one learned scalar.",
    shape="x[B, ...] -> out[B, ...]",
    args={
        "init_value": Arg(float, 0.0, positional=False,
                          help="Value the scalar gamma starts at; 0 makes the layer start as a zero map, "
                               "the SAGAN convention for a gated residual branch."),
    },
    reference=lambda module: (lambda x: x * module.gamma),
    examples=[
        Example("saved = x\nconv(4, kernel_size=3, padding=1)\nrelu()\nh = conv(4, kernel_size=1)\n"
                "s = learned_scale(h)\nadd(s, saved)", ("B", 4, 8, 8), ("B", 4, 8, 8),
                "The SAGAN gate: gamma starts at 0, so the branch begins as the identity and learns its weight."),
        Example("linear(8)\nlearned_scale(init_value=1.0)\ntanh()", ("B", 5, 4), ("B", 5, 8),
                "Sequences: one scalar gain, shared by every position and feature, starting at 1."),
        Example("linear()\nlearned_scale(init_value=0.5)", ("B", 16), ("B", 10),
                "A learned gain on a projection; the width 10 still flows back through it."),
    ],
    category="arithmetic",
)
class LearnedScale(nn.Module):
    """``out = gamma * x`` where ``gamma`` is a single learned scalar shared by
    every element of the tensor:

    ```text
    out[i, ...] = gamma * x[i, ...]
    ```

    The one parameter is named ``gamma`` and has shape ``[1]``, so it is the
    target of construction overrides by that exact name:
    ``learned_scale(init={"gamma": 1.0})`` sets it and
    ``learned_scale(trainable={"gamma": False})`` freezes it. ``init_value``
    gives the value the constructor starts from when no initialization
    override applies; it is a float and does not take part in shape inference.

    The default ``init_value=0.0`` is the Self-Attention GAN convention (Zhang
    et al. 2018): the layer starts as a zero map, so a residual
    ``add(learned_scale(branch), saved)`` begins exactly as the identity and
    the network learns how much of the branch to admit. Note that a gamma of
    zero also zeroes the gradient flowing back through ``x``; the branch
    parameters start learning through ``gamma`` itself, whose gradient is the
    inner product of the branch output with the incoming gradient.

    Rank, shape, and dtype are preserved, there are no buffers, and train and
    eval behave identically. Use ``scale`` instead for a fixed constant that is
    not learned.
    """

    def __init__(self, init_value):
        super().__init__()
        self.init_value = float(init_value)
        self.gamma = nn.Parameter(torch.full((1,), self.init_value))

    def forward(self, x):
        return self.gamma * x

    def extra_repr(self):
        return f"init_value={self.init_value}"
