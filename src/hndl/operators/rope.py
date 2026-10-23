"""Standalone rotary position embedding over a [B, T, D] sequence."""

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _rotate_half(tensor):
    half = tensor.shape[-1] // 2
    left, right = tensor[..., :half], tensor[..., half:]
    return torch.cat((-right, left), dim=-1)


def _relation(s):
    s.rank("x", 3)
    s.rank("out", 3)
    heads = s.args.get("heads", 1)
    a, b = s.shape("x"), s.shape("out")
    if a is not None:
        if a[1] is not None:
            s.axis("out", 1, a[1])
        if a[2] is not None:
            s.axis("out", 2, a[2])
    if b is not None:
        if b[1] is not None:
            s.axis("x", 1, b[1])
        if b[2] is not None:
            s.axis("x", 2, b[2])
    shape = s.shape("x") or s.shape("out")
    if shape is not None and shape[2] is not None:
        d = shape[2]
        if d % heads != 0:
            s.error("E_CONSTRAINT", f"rope heads={heads} must divide width D={d}")
        head_dim = d // heads
        if head_dim % 2 != 0:
            s.error("E_CONSTRAINT", f"rope head_dim={head_dim} (D // heads) must be even")


def _reference(module):
    heads = module.heads
    theta = module.theta

    def run(x):
        b, t, d = x.shape
        dh = d // heads
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        exponent = torch.arange(0, dh, 2, device=x.device, dtype=calc_dtype) / dh
        inverse_frequency = theta ** (-exponent)
        angles = torch.arange(t, device=x.device, dtype=calc_dtype)[:, None] * inverse_frequency[None, :]
        duplicated = torch.cat((angles, angles), dim=-1)
        cos = duplicated.cos().to(x.dtype)[None, :, None, :]
        sin = duplicated.sin().to(x.dtype)[None, :, None, :]
        xh = x.view(b, t, heads, dh)
        half = dh // 2
        rot_half = torch.cat((-xh[..., half:], xh[..., :half]), dim=-1)
        out = xh * cos + rot_half * sin
        return out.view(b, t, d)

    return run


@operator(
    "rope",
    summary="Standalone rotary position embedding over a [B, T, D] sequence.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    shape_text="D % heads == 0 and (D // heads) % 2 == 0; out matches x",
    args={
        "heads": Arg(int, 1, min=1, help="Number of heads to split D into before rotating."),
        "theta": Arg(
            float,
            10000.0,
            min=1.0,
            exclusive_min=True,
            positional=False,
            help="Base frequency for rotary embeddings.",
        ),
    },
    examples=[
        Example(
            "linear(64)\nrope(4)\nlinear()",
            ("B", 16, 32),
            ("B", 16, 10),
            "Rotates 4 heads of dimension 16 along the length axis.",
        ),
    ],
    reference=_reference,
    category="sequence",
)
class RoPE(nn.Module):
    """Standalone rotary position embedding (RoPE) for [B, T, D] sequences.

    Applies rotary position embeddings independently to each head of dimension
    ``head_dim = D // heads`` along the sequence length axis:
    ``out = x * cos + rotate_half(x) * sin``.
    """

    def __init__(self, heads=1, theta=10000.0):
        super().__init__()
        self.heads = int(heads)
        self.theta = float(theta)
        self._cache = {}

    def _apply(self, fn, recurse=True):
        res = super()._apply(fn, recurse=recurse)
        self._cache.clear()
        return res

    def _tables(self, positions, head_dim, device, dtype):
        if torch.compiler.is_compiling():
            calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            exponent = torch.arange(0, head_dim, 2, device=device, dtype=calc_dtype) / head_dim
            inverse_frequency = self.theta ** (-exponent)
            angles = torch.arange(positions, device=device, dtype=calc_dtype)[:, None] * inverse_frequency[None, :]
            duplicated = torch.cat((angles, angles), dim=-1)
            cos = duplicated.cos().to(dtype)[None, :, None, :]
            sin = duplicated.sin().to(dtype)[None, :, None, :]
            return cos, sin
        key = (head_dim, str(device), dtype)
        hit = self._cache.get(key)
        if hit is not None and hit[0].shape[1] >= positions:
            return hit[0][:, :positions, :, :], hit[1][:, :positions, :, :]
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        exponent = torch.arange(0, head_dim, 2, device=device, dtype=calc_dtype) / head_dim
        inverse_frequency = self.theta ** (-exponent)
        angles = torch.arange(positions, device=device, dtype=calc_dtype)[:, None] * inverse_frequency[None, :]
        duplicated = torch.cat((angles, angles), dim=-1)
        cos = duplicated.cos().to(dtype)[None, :, None, :]
        sin = duplicated.sin().to(dtype)[None, :, None, :]
        self._cache[key] = (cos, sin)
        return cos, sin

    def forward(self, x):
        b, t, d = x.shape
        dh = d // self.heads
        cos, sin = self._tables(t, dh, x.device, x.dtype)
        xh = x.view(b, t, self.heads, dh)
        rotated = xh * cos + _rotate_half(xh) * sin
        return rotated.view(b, t, d)

    def extra_repr(self):
        return f"heads={self.heads}, theta={self.theta}"
