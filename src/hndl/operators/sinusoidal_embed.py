"""Deterministic log-spaced sinusoidal timestep embeddings."""

import math

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    s.rank("x", 2)
    s.rank("out", 2)
    features = s.args.get("features")
    x_shape = s.shape("x")
    out_shape = s.shape("out")
    if features is not None:
        s.axis("out", 1, features)
    elif out_shape is not None and out_shape[1] is not None:
        s.arg("features", out_shape[1])
        features = out_shape[1]
    if features is not None:
        if features % 2 != 0:
            s.error("E_CONSTRAINT", f"sinusoidal_embed features must be even (got {features})")
        if x_shape is not None and x_shape[1] is not None:
            d_in = x_shape[1]
            if features % (2 * d_in) != 0:
                s.error(
                    "E_CONSTRAINT",
                    f"sinusoidal_embed features ({features}) must be a multiple of 2 * D_in ({2 * d_in})",
                )


def _reference(module):
    features, max_period, d_in = module.features, module.max_period, module.D_in
    d_per = features // (2 * d_in)

    def run(x):
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_calc = x.to(calc_dtype)
        if d_per > 1:
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(d_per, device=x.device, dtype=calc_dtype) / (d_per - 1)
            )
        else:
            freqs = torch.ones(1, device=x.device, dtype=calc_dtype)
        angles = x_calc.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.view(x.shape[0], features).to(x.dtype)

    return run


@operator(
    "sinusoidal_embed",
    summary="Deterministic log-spaced sinusoidal timestep embeddings.",
    shape="x[B, D_in] -> out[B, D_out]",
    relation=_relation,
    shape_text="D_out = features (must be even and a multiple of 2 * D_in)",
    args={
        "features": Arg(int, inferable=True, min=1, dim="D_out", help="Output embedding dimension."),
        "max_period": Arg(
            float,
            10000.0,
            min=1.0,
            exclusive_min=True,
            positional=False,
            help="Maximum frequency period for log-spaced frequencies.",
        ),
    },
    examples=[
        Example(
            "sinusoidal_embed(64)",
            ("B", 1),
            ("B", 64),
            "Embeds scalar timesteps into 64 sinusoidal features.",
        ),
    ],
    reference=_reference,
    category="sequence",
)
class SinusoidalEmbed(nn.Module):
    """Deterministic log-spaced sinusoidal timestep and coordinate embedding.

    Computes frequency bands geometrically spaced from 1 down to 1 / max_period,
    mapping inputs into interleaved or concatenated sine and cosine components:
    ``angles = x * freqs``
    ``out = [sin(angles), cos(angles)]``.
    """

    def __init__(self, features, max_period=10000.0, *, D_in):
        super().__init__()
        self.features = int(features)
        self.max_period = float(max_period)
        self.D_in = int(D_in)
        d_per = self.features // (2 * self.D_in)
        if d_per > 1:
            freqs = torch.exp(
                -math.log(self.max_period) * torch.arange(d_per, dtype=torch.float32) / (d_per - 1)
            )
        else:
            freqs = torch.ones(1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if self.freqs.dtype == calc_dtype and self.freqs.device == x.device:
            freqs = self.freqs
        else:
            d_per = self.features // (2 * self.D_in)
            if d_per > 1:
                freqs = torch.exp(
                    -math.log(self.max_period)
                    * torch.arange(d_per, device=x.device, dtype=calc_dtype)
                    / (d_per - 1)
                )
            else:
                freqs = torch.ones(1, device=x.device, dtype=calc_dtype)
        angles = x.to(calc_dtype).unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.view(x.shape[0], self.features).to(x.dtype)

    def extra_repr(self):
        return f"features={self.features}, max_period={self.max_period}"

