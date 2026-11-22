"""Periodic activation (Snake) operator for neural audio synthesis."""

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        return
    rank = len(shape)
    s.rank("x", rank)
    s.rank("out", rank)
    s.equal("x", "out")

    channels = s.args.get("channels", 0)
    dim = s.args.get("dim", 0)
    c = channels if channels > 0 else dim
    if c == 0:
        inferred = shape[1] if rank == 4 else shape[-1]
        if inferred is not None:
            s.args["channels"] = inferred
            s._solver.changed = True
            c = inferred
    if c > 0:
        if s.args.get("channels", 0) != c:
            s.args["channels"] = c
            s._solver.changed = True
        if rank == 2:
            s.axis("x", 1, c)
            s.axis("out", 1, c)
        else:
            dim1 = shape[1]
            dim_last = shape[-1]
            preferred_axis = 1 if rank == 4 else rank - 1
            alt_axis = rank - 1 if rank == 4 else 1
            preferred_dim = shape[preferred_axis]
            alt_dim = shape[alt_axis]
            if preferred_dim == c:
                s.axis("x", preferred_axis, c)
                s.axis("out", preferred_axis, c)
            elif alt_dim == c:
                s.axis("x", alt_axis, c)
                s.axis("out", alt_axis, c)
            elif dim1 is not None and dim_last is not None:
                s.error(
                    "E_CONSTRAINT",
                    f"snake channels={c} does not match input channel dimension "
                    f"(dim 1 is {dim1}, dim -1 is {dim_last})",
                )
            elif preferred_dim is not None:
                s.axis("x", alt_axis, c)
                s.axis("out", alt_axis, c)
            else:
                s.axis("x", preferred_axis, c)
                s.axis("out", preferred_axis, c)


def _alpha_view(alpha, c_val, x):
    if c_val <= 0:
        return alpha
    preferred = 1 if x.ndim == 4 else x.ndim - 1
    alt = x.ndim - 1 if x.ndim == 4 else 1
    if x.shape[preferred] == c_val:
        view_shp = [1] * x.ndim
        view_shp[preferred] = c_val
        return alpha.view(view_shp)
    if x.shape[alt] == c_val:
        view_shp = [1] * x.ndim
        view_shp[alt] = c_val
        return alpha.view(view_shp)
    return alpha


def _reference(module):
    alpha = module.alpha
    eps = module.eps
    c_val = module.channels

    def run(x):
        a = _alpha_view(alpha, c_val, x)
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_calc = x.to(calc_dtype)
        a_calc = a.to(calc_dtype)
        denom = a_calc + eps
        return (x_calc + (torch.sin(a_calc * x_calc).square() / denom)).to(x.dtype)

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
        a = _alpha_view(self.alpha, self.channels, x)
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_calc = x.to(calc_dtype)
        a_calc = a.to(calc_dtype)
        denom = a_calc + self.eps
        return (x_calc + (torch.sin(a_calc * x_calc).square() / denom)).to(x.dtype)

    def extra_repr(self):
        return f"channels={self.channels}, eps={self.eps}"
