"""Standalone rotary position embedding over a [B, T, D] or [B, H, T, D_h] sequence."""

import math
import torch
from torch import nn

from ..operator import PAIR, Arg, Example, operator


def _rotate_half(tensor):
    half = tensor.shape[-1] // 2
    left, right = tensor[..., :half], tensor[..., half:]
    return torch.cat((-right, left), dim=-1)


def compute_rope_tables(
    positions,
    head_dim,
    device,
    dtype,
    theta=10000.0,
    rope_scaling="none",
    rope_scale=1.0,
    spatial_shape=(0, 0),
):
    """Compute cos and sin rotary tables with optional scaling and 2D axial grid."""
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    sh = spatial_shape
    if isinstance(sh, int):
        sh = (sh, sh)
    h_grid, w_grid = int(sh[0]), int(sh[1])

    if h_grid > 0 and w_grid > 0:
        d_axis = head_dim // 2
        exp = torch.arange(0, d_axis, 2, device=device, dtype=calc_dtype) / d_axis
        base_freq = float(theta) ** (-exp)
        if rope_scaling == "linear":
            base_freq = base_freq / float(rope_scale)
        elif rope_scaling == "ntk":
            theta_ntk = float(theta) * (float(rope_scale) ** (head_dim / max(head_dim - 2, 1)))
            base_freq = float(theta_ntk) ** (-exp)
        elif rope_scaling == "yarn":
            wavelength = 2.0 * math.pi / (base_freq + 1e-12)
            low_bound = float(positions) / 32.0
            high_bound = float(positions) / 1.0
            gamma = torch.clamp((high_bound - wavelength) / (high_bound - low_bound + 1e-8), 0.0, 1.0)
            base_freq = (1.0 - gamma) * (base_freq / float(rope_scale)) + gamma * base_freq
        elif rope_scaling == "llama3":
            wavelength = 2.0 * math.pi / (base_freq + 1e-12)
            low_bound = 8192.0 / 4.0
            high_bound = 8192.0 / 1.0
            gamma = torch.clamp((wavelength - low_bound) / (high_bound - low_bound + 1e-8), 0.0, 1.0)
            smooth = gamma * gamma * (3.0 - 2.0 * gamma)
            base_freq = (1.0 - smooth) * base_freq + smooth * (base_freq / float(rope_scale))

        token_indices = torch.arange(positions, device=device, dtype=calc_dtype)
        rows = token_indices // w_grid
        cols = token_indices % w_grid
        angles_h = rows[:, None] * base_freq[None, :]
        angles_w = cols[:, None] * base_freq[None, :]
        angles_2d = torch.cat((angles_h, angles_w), dim=-1)
        duplicated = torch.cat((angles_2d, angles_2d), dim=-1)
        return duplicated.cos().to(dtype)[None, None, :, :], duplicated.sin().to(dtype)[None, None, :, :]

    exponent = torch.arange(0, head_dim, 2, device=device, dtype=calc_dtype) / head_dim
    base_freq = float(theta) ** (-exponent)
    if rope_scaling == "linear":
        base_freq = base_freq / float(rope_scale)
    elif rope_scaling == "ntk":
        theta_ntk = float(theta) * (float(rope_scale) ** (head_dim / max(head_dim - 2, 1)))
        base_freq = float(theta_ntk) ** (-exponent)
    elif rope_scaling == "yarn":
        wavelength = 2.0 * math.pi / (base_freq + 1e-12)
        low_bound = float(positions) / 32.0
        high_bound = float(positions) / 1.0
        gamma = torch.clamp((high_bound - wavelength) / (high_bound - low_bound + 1e-8), 0.0, 1.0)
        base_freq = (1.0 - gamma) * (base_freq / float(rope_scale)) + gamma * base_freq
    elif rope_scaling == "llama3":
        wavelength = 2.0 * math.pi / (base_freq + 1e-12)
        low_bound = 8192.0 / 4.0
        high_bound = 8192.0 / 1.0
        gamma = torch.clamp((wavelength - low_bound) / (high_bound - low_bound + 1e-8), 0.0, 1.0)
        smooth = gamma * gamma * (3.0 - 2.0 * gamma)
        base_freq = (1.0 - smooth) * base_freq + smooth * (base_freq / float(rope_scale))

    angles = torch.arange(positions, device=device, dtype=calc_dtype)[:, None] * base_freq[None, :]
    duplicated = torch.cat((angles, angles), dim=-1)
    return duplicated.cos().to(dtype)[None, None, :, :], duplicated.sin().to(dtype)[None, None, :, :]


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    heads = s.args.get("heads", 1)
    sp = s.args.get("spatial_shape", (0, 0))
    if isinstance(sp, int):
        sp = (sp, sp)
    sh, sw = sp

    if shape is not None and len(shape) == 4:
        s.rank("x", 4)
        s.rank("out", 4)
        a, b = s.shape("x"), s.shape("out")
        if a is not None:
            for i in range(1, 4):
                if a[i] is not None:
                    s.axis("out", i, a[i])
        if b is not None:
            for i in range(1, 4):
                if b[i] is not None:
                    s.axis("x", i, b[i])
        shape = s.shape("x") or s.shape("out")
        if shape is not None:
            if shape[1] is not None and heads > 1:
                s.axis("x", 1, heads)
                s.axis("out", 1, heads)
            if shape[3] is not None:
                dh = shape[3]
                if dh % 2 != 0:
                    s.error("E_CONSTRAINT", f"rope head_dim={dh} must be even")
                if (sh > 0 or sw > 0) and dh % 4 != 0:
                    s.error("E_CONSTRAINT", f"rope 2D axial requires head_dim divisible by 4 (got {dh})")
            if shape[2] is not None and (sh > 0 or sw > 0):
                t = shape[2]
                if sh <= 0 or sw <= 0:
                    s.error("E_ARGUMENT", "spatial_shape must have positive height and width")
                if t != sh * sw:
                    s.error("E_CONSTRAINT", f"rope spatial_shape={sp} covers {sh * sw} tokens, but sequence length is {t}")
    else:
        s.rank("x", 3)
        s.rank("out", 3)
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
        if shape is not None:
            if shape[2] is not None:
                d = shape[2]
                if d % heads != 0:
                    s.error("E_CONSTRAINT", f"rope heads={heads} must divide width D={d}")
                head_dim = d // heads
                if head_dim % 2 != 0:
                    s.error("E_CONSTRAINT", f"rope head_dim={head_dim} (D // heads) must be even")
                if (sh > 0 or sw > 0) and head_dim % 4 != 0:
                    s.error("E_CONSTRAINT", f"rope 2D axial requires head_dim divisible by 4 (got {head_dim})")
            if shape[1] is not None and (sh > 0 or sw > 0):
                t = shape[1]
                if sh <= 0 or sw <= 0:
                    s.error("E_ARGUMENT", "spatial_shape must have positive height and width")
                if t != sh * sw:
                    s.error("E_CONSTRAINT", f"rope spatial_shape={sp} covers {sh * sw} tokens, but sequence length is {t}")


def _reference(module):
    heads = module.heads
    theta = module.theta
    rope_scaling = getattr(module, "rope_scaling", "none")
    rope_scale = getattr(module, "rope_scale", 1.0)
    spatial_shape = getattr(module, "spatial_shape", (0, 0))

    def run(x):
        if x.ndim == 4:
            b, h, t, dh = x.shape
            cos, sin = compute_rope_tables(
                t, dh, x.device, x.dtype,
                theta=theta, rope_scaling=rope_scaling, rope_scale=rope_scale, spatial_shape=spatial_shape,
            )
            return x * cos + _rotate_half(x) * sin
        b, t, d = x.shape
        dh = d // heads
        cos, sin = compute_rope_tables(
            t, dh, x.device, x.dtype,
            theta=theta, rope_scaling=rope_scaling, rope_scale=rope_scale, spatial_shape=spatial_shape,
        )
        xh = x.view(b, t, heads, dh)
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        out = xh * cos + _rotate_half(xh) * sin
        return out.view(b, t, d)

    return run


@operator(
    "rope",
    summary="Standalone rotary position embedding over a [B, T, D] or [B, H, T, D_h] sequence.",
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
        "rope_scaling": Arg(
            str,
            "none",
            choices=("none", "linear", "ntk", "yarn", "llama3"),
            positional=False,
            since="0.10.0",
            help="Frequency scaling algorithm for RoPE context extension.",
        ),
        "rope_scale": Arg(
            float,
            1.0,
            min=0.0,
            exclusive_min=True,
            positional=False,
            since="0.10.0",
            help="Frequency scale factor for context extension.",
        ),
        "spatial_shape": Arg(
            PAIR,
            (0, 0),
            min=0,
            positional=False,
            since="0.10.0",
            help="Token grid (H, W) for 2D axial RoPE, row-major with H*W == T.",
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
    """Standalone rotary position embedding (RoPE) for [B, T, D] or [B, H, T, D_h] sequences.

    Applies rotary position embeddings independently to each head of dimension
    ``head_dim = D // heads`` along the sequence length axis:
    ``out = x * cos + rotate_half(x) * sin``.
    """

    def __init__(self, heads=1, theta=10000.0, rope_scaling="none", rope_scale=1.0, spatial_shape=(0, 0)):
        super().__init__()
        self.heads = int(heads)
        self.theta = float(theta)
        self.rope_scaling = str(rope_scaling)
        self.rope_scale = float(rope_scale)
        sh = spatial_shape
        if isinstance(sh, int):
            sh = (sh, sh)
        self.spatial_shape = (int(sh[0]), int(sh[1]))
        self._cache = {}

    def _apply(self, fn, recurse=True):
        res = super()._apply(fn, recurse=recurse)
        self._cache.clear()
        return res

    def _tables(self, positions, head_dim, device, dtype):
        if torch.compiler.is_compiling():
            c, s = compute_rope_tables(
                positions, head_dim, device, dtype,
                theta=self.theta, rope_scaling=self.rope_scaling,
                rope_scale=self.rope_scale, spatial_shape=self.spatial_shape,
            )
            return c.transpose(1, 2), s.transpose(1, 2)
        key = (head_dim, str(device), dtype)
        hit = self._cache.get(key)
        if hit is not None and hit[0].shape[1] >= positions:
            return hit[0][:, :positions, :, :], hit[1][:, :positions, :, :]
        cos, sin = compute_rope_tables(
            positions, head_dim, device, dtype,
            theta=self.theta, rope_scaling=self.rope_scaling,
            rope_scale=self.rope_scale, spatial_shape=self.spatial_shape,
        )
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        self._cache[key] = (cos, sin)
        return cos, sin

    def forward(self, x):
        if x.ndim == 4:
            b, h, t, dh = x.shape
            cos, sin = self._tables(t, dh, x.device, x.dtype)
            cos = cos.transpose(1, 2)
            sin = sin.transpose(1, 2)
            rotated = x * cos + _rotate_half(x) * sin
            return rotated
        b, t, d = x.shape
        dh = d // self.heads
        cos, sin = self._tables(t, dh, x.device, x.dtype)
        xh = x.view(b, t, self.heads, dh)
        rotated = xh * cos + _rotate_half(xh) * sin
        return rotated.view(b, t, d)

    def extra_repr(self):
        extras = [f"heads={self.heads}", f"theta={self.theta}"]
        if self.rope_scaling != "none":
            extras.append(f"rope_scaling='{self.rope_scaling}'")
        if self.rope_scale != 1.0:
            extras.append(f"rope_scale={self.rope_scale}")
        if self.spatial_shape != (0, 0):
            extras.append(f"spatial_shape={self.spatial_shape}")
        return ", ".join(extras)

