"""Periodic activation (Snake) operator for neural audio synthesis."""

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        rank = 3
    else:
        rank = len(shape)
    s.rank("x", rank)
    s.rank("out", rank)
    s.equal("x", "out")

    channels = s.args.get("channels", 0)
    dim = s.args.get("dim", 0)
    c = channels if channels > 0 else dim
    if c > 0:
        s.args["channels"] = c
        if shape is not None:
            if rank == 2:
                s.axis("x", 1, c)
                s.axis("out", 1, c)
            else:
                dim1 = shape[1]
                dim_last = shape[-1]
                if dim1 == c:
                    s.axis("x", 1, c)
                    s.axis("out", 1, c)
                elif dim_last == c:
                    s.axis("x", -1, c)
                    s.axis("out", -1, c)
                elif dim1 is not None and dim_last is not None:
                    s.error(
                        "E_CONSTRAINT",
                        f"snake channels={c} does not match input channel dimension "
                        f"(dim 1 is {dim1}, dim -1 is {dim_last})",
                    )
                elif dim1 is not None:
                    s.axis("x", -1, c)
                    s.axis("out", -1, c)
                elif dim_last is not None:
                    s.axis("x", 1, c)
                    s.axis("out", 1, c)
                else:
                    s.axis("x", 1, c)
                    s.axis("out", 1, c)


def _reference(module):
    alpha = module.alpha
    eps = module.eps
    c_val = module.channels

    def run(x):
        if c_val > 0:
            if x.shape[1] == c_val:
                view_shp = [1] * x.ndim
                view_shp[1] = c_val
                a = alpha.view(view_shp)
            elif x.shape[-1] == c_val:
                view_shp = [1] * x.ndim
                view_shp[-1] = c_val
                a = alpha.view(view_shp)
            else:
                a = alpha
        else:
            a = alpha
        denom = a + eps
        return x + (1.0 / denom) * (torch.sin(a * x) ** 2)

    return run


@operator(
    "snake",
    identity="snake",
    summary="Periodic activation: x + (1 / (alpha + eps)) * sin^2(alpha * x).",
    shape="x[B, ...] -> out[B, ...]",
    relation=_relation,
    args={
        "channels": Arg(int, 0, min=0, help="Number of channels C; 0 infers from input."),
        "dim": Arg(int, 0, min=0, positional=False, help="Alias for channels."),
        "alpha": Arg(float, 1.0, positional=False, help="Initial frequency scale parameter alpha."),
        "eps": Arg(float, 1e-9, min=0.0, exclusive_min=True, positional=False, help="Epsilon to prevent division by zero."),
    },
    examples=[
        Example(
            "snake(16)",
            ("B", 16, 64),
            ("B", 16, 64),
            "Periodic activation with learnable frequency parameter per channel.",
        ),
    ],
    reference=_reference,
    category="activation",
)
class Snake(nn.Module):
    """Periodic activation function for neural audio and inductive bias."""

    def __init__(self, channels=0, dim=0, alpha=1.0, eps=1e-9, *, C=0, D=0):
        super().__init__()
        c = channels if channels > 0 else (dim if dim > 0 else (C if C > 0 else D))
        self.channels = int(c)
        self.eps = float(eps)
        if self.channels > 0:
            self.alpha = nn.Parameter(torch.full((self.channels,), float(alpha)))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x):
        if self.channels > 0:
            if x.shape[1] == self.channels:
                view_shp = [1] * x.ndim
                view_shp[1] = self.channels
                a = self.alpha.view(view_shp)
            elif x.shape[-1] == self.channels:
                view_shp = [1] * x.ndim
                view_shp[-1] = self.channels
                a = self.alpha.view(view_shp)
            else:
                a = self.alpha
        else:
            a = self.alpha
        denom = a + self.eps
        return x + (1.0 / denom) * (torch.sin(a * x) ** 2)

    def extra_repr(self):
        return f"channels={self.channels}, eps={self.eps}"
