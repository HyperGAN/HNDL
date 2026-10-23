"""State Space Duality (SSD / Mamba-2) operator."""

import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _relation(s):
    s.rank("x", 3)
    s.rank("out", 3)
    d_model = s.args.get("d_model", 0)
    expand = s.args.get("expand", 2)
    heads = s.args.get("heads", 8)
    head_dim = s.args.get("head_dim", 64)

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
        if d_model == 0:
            d_model = shape[2]
            s.axis("x", 2, d_model)
            s.axis("out", 2, d_model)
        elif shape[2] != d_model:
            s.error("E_CONSTRAINT", f"mamba2 d_model={d_model} does not match tensor width {shape[2]}")

    if d_model > 0:
        inner_dim = expand * d_model
        if heads * head_dim != inner_dim:
            s.error(
                "E_CONSTRAINT",
                f"mamba2 heads * head_dim ({heads} * {head_dim} = {heads * head_dim}) "
                f"must equal inner_dim (expand * d_model = {expand} * {d_model} = {inner_dim})",
            )


def _reference(module):
    def run(x):
        b, t, d = x.shape
        inner_dim = module.heads * module.head_dim

        # Input projection
        zxbcdt = module.in_proj(x)
        z = zxbcdt[..., :inner_dim]
        x_conv = zxbcdt[..., inner_dim : 2 * inner_dim]
        b_proj = zxbcdt[..., 2 * inner_dim : 2 * inner_dim + module.heads * module.d_state]
        c_proj = zxbcdt[
            ...,
            2 * inner_dim + module.heads * module.d_state : 2 * inner_dim + 2 * module.heads * module.d_state,
        ]
        dt_raw = zxbcdt[..., -module.heads :]

        # 1D causal depthwise conv on x_conv
        x_conv_t = x_conv.transpose(1, 2)
        conv_out = module.conv1d(x_conv_t)[..., :t]
        x_conv = F.silu(conv_out).transpose(1, 2)

        # Reshape to heads
        x_h = x_conv.view(b, t, module.heads, module.head_dim)
        b_h = b_proj.view(b, t, module.heads, module.d_state)
        c_h = c_proj.view(b, t, module.heads, module.d_state)
        dt = F.softplus(dt_raw + module.dt_bias)

        # Discretize A
        a_scaled = -torch.exp(module.A_log)
        a_bar = torch.exp(dt * a_scaled)  # [B, T, H]

        # SSD Recurrence
        # state: [B, H, P, N]
        state = torch.zeros(
            b, module.heads, module.head_dim, module.d_state, device=x.device, dtype=x.dtype
        )
        y_list = []
        for i in range(t):
            xt = x_h[:, i]  # [B, H, P]
            bt = b_h[:, i]  # [B, H, N]
            ct = c_h[:, i]  # [B, H, N]
            decay = a_bar[:, i, :, None, None]  # [B, H, 1, 1]
            state = state * decay + xt[..., :, None] * bt[..., None, :]
            yt = (state * ct[..., None, :]).sum(dim=-1)  # [B, H, P]
            if module.D is not None:
                yt = yt + xt * module.D[None, :, None]
            y_list.append(yt)

        y = torch.stack(y_list, dim=1).view(b, t, inner_dim)
        y = module.norm(y) * F.silu(z)
        out = module.out_proj(y)
        return out

    return run


@operator(
    "mamba2",
    summary="State Space Duality (SSD) block from Mamba-2.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    args={
        "d_model": Arg(int, inferable=True, dim="D", min=1, help="Model feature dimension D."),
        "d_state": Arg(int, 64, min=1, positional=False, help="State space dimension N per head."),
        "d_conv": Arg(int, 4, min=1, positional=False, help="1D depthwise convolution width."),
        "expand": Arg(int, 2, min=1, positional=False, help="Inner dimension expansion factor."),
        "heads": Arg(int, 8, min=1, positional=False, help="Number of heads."),
        "head_dim": Arg(int, 64, min=1, positional=False, help="Head dimension."),
        "bias": Arg(bool, False, positional=False, help="Whether linear projections have bias."),
        "conv_bias": Arg(bool, True, positional=False, help="Whether depthwise conv1d has bias."),
        "d_has_bias": Arg(bool, False, positional=False, help="Whether skip connection D has bias."),
    },
    examples=[
        Example(
            "mamba2(d_model=32, d_state=16, heads=4, head_dim=16)",
            ("B", 8, 32),
            ("B", 8, 32),
            "Mamba-2 SSD sequence transformation block.",
        ),
    ],
    reference=_reference,
    category="sequence",
)
class Mamba2(nn.Module):
    """Mamba-2 State Space Duality (SSD) layer."""

    def __init__(
        self,
        d_model=0,
        d_state=64,
        d_conv=4,
        expand=2,
        heads=8,
        head_dim=64,
        bias=False,
        conv_bias=True,
        d_has_bias=False,
        *,
        D=0,
    ):
        super().__init__()
        actual_d_model = int(d_model) if d_model > 0 else int(D)
        self.d_model = actual_d_model
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.heads * self.head_dim
        self.bias = bool(bias)

        in_proj_dim = 2 * self.inner_dim + 2 * self.heads * self.d_state + self.heads
        self.in_proj = nn.Linear(self.d_model, in_proj_dim, bias=self.bias)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=self.d_conv,
            groups=self.inner_dim,
            padding=self.d_conv - 1,
            bias=bool(conv_bias),
        )

        self.A_log = nn.Parameter(torch.empty(self.heads))
        nn.init.uniform_(self.A_log, 0.0, 3.0)

        self.dt_bias = nn.Parameter(torch.empty(self.heads))
        nn.init.constant_(self.dt_bias, 0.0)

        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = nn.RMSNorm(self.inner_dim, eps=1e-5)
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=self.bias)

    def forward(self, x):
        b, t, d = x.shape
        zxbcdt = self.in_proj(x)
        z = zxbcdt[..., : self.inner_dim]
        x_conv = zxbcdt[..., self.inner_dim : 2 * self.inner_dim]
        b_proj = zxbcdt[
            ...,
            2 * self.inner_dim : 2 * self.inner_dim + self.heads * self.d_state,
        ]
        c_proj = zxbcdt[
            ...,
            2 * self.inner_dim
            + self.heads * self.d_state : 2 * self.inner_dim
            + 2 * self.heads * self.d_state,
        ]
        dt_raw = zxbcdt[..., -self.heads :]

        x_conv_t = x_conv.transpose(1, 2)
        conv_out = self.conv1d(x_conv_t)[..., :t]
        x_conv = F.silu(conv_out).transpose(1, 2)

        x_h = x_conv.view(b, t, self.heads, self.head_dim)
        b_h = b_proj.view(b, t, self.heads, self.d_state)
        c_h = c_proj.view(b, t, self.heads, self.d_state)
        dt = F.softplus(dt_raw + self.dt_bias)

        a_scaled = -torch.exp(self.A_log)
        a_bar = torch.exp(dt * a_scaled)

        state = torch.zeros(
            b, self.heads, self.head_dim, self.d_state, device=x.device, dtype=x.dtype
        )
        y_list = []
        for i in range(t):
            xt = x_h[:, i]
            bt = b_h[:, i]
            ct = c_h[:, i]
            decay = a_bar[:, i, :, None, None]
            state = state * decay + xt[..., :, None] * bt[..., None, :]
            yt = (state * ct[..., None, :]).sum(dim=-1)
            if self.D is not None:
                yt = yt + xt * self.D[None, :, None]
            y_list.append(yt)

        y = torch.stack(y_list, dim=1).view(b, t, self.inner_dim)
        y = self.norm(y) * F.silu(z)
        out = self.out_proj(y)
        return out

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, heads={self.heads}, "
            f"head_dim={self.head_dim}, expand={self.expand}"
        )
