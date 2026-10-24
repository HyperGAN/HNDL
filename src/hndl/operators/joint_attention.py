"""Dual-stream Multimodal Diffusion Transformer (MMDiT) joint attention operator."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _relation(s):
    s.rank("x", 3)
    s.rank("context", 3)
    s.rank("out", 3)
    s.rank("ctx_out", 3)

    heads = s.args.get("heads", 1)
    head_dim = s.args.get("head_dim", 0)

    # Match batch dimensions
    s.share_batch("x", "context", "out", "ctx_out")

    # Match sequence and feature dimensions
    x_shp = s.shape("x")
    if x_shp is not None:
        if x_shp[1] is not None:
            s.axis("out", 1, x_shp[1])
        if x_shp[2] is not None:
            s.axis("out", 2, x_shp[2])
    out_shp = s.shape("out")
    if out_shp is not None:
        if out_shp[1] is not None:
            s.axis("x", 1, out_shp[1])
        if out_shp[2] is not None:
            s.axis("x", 2, out_shp[2])

    c_shp = s.shape("context")
    if c_shp is not None:
        if c_shp[1] is not None:
            s.axis("ctx_out", 1, c_shp[1])
        if c_shp[2] is not None:
            s.axis("ctx_out", 2, c_shp[2])
    ctx_out_shp = s.shape("ctx_out")
    if ctx_out_shp is not None:
        if ctx_out_shp[1] is not None:
            s.axis("context", 1, ctx_out_shp[1])
        if ctx_out_shp[2] is not None:
            s.axis("context", 2, ctx_out_shp[2])

    shp = s.shape("x") or s.shape("out")
    if shp is not None and shp[2] is not None and head_dim == 0:
        d = shp[2]
        if d % heads != 0:
            s.error("E_CONSTRAINT", f"joint_attention heads={heads} must divide width D={d}")


def _reference(module):
    heads = module.heads
    head_dim = module.head_dim
    dropout = module.dropout
    qk_norm = module.qk_norm

    def run(x, context):
        b, tx, _ = x.shape
        _, tc, _ = context.shape

        qx = module.x_q_proj(x).view(b, tx, heads, head_dim).transpose(1, 2)
        kx = module.x_k_proj(x).view(b, tx, heads, head_dim).transpose(1, 2)
        vx = module.x_v_proj(x).view(b, tx, heads, head_dim).transpose(1, 2)

        qc = module.c_q_proj(context).view(b, tc, heads, head_dim).transpose(1, 2)
        kc = module.c_k_proj(context).view(b, tc, heads, head_dim).transpose(1, 2)
        vc = module.c_v_proj(context).view(b, tc, heads, head_dim).transpose(1, 2)

        if qk_norm == "rms_norm":
            qx = module.x_q_norm(qx)
            kx = module.x_k_norm(kx)
            qc = module.c_q_norm(qc)
            kc = module.c_k_norm(kc)
        elif qk_norm == "l2_norm":
            qx = F.normalize(qx, p=2.0, dim=-1, eps=module.eps)
            kx = F.normalize(kx, p=2.0, dim=-1, eps=module.eps)
            qc = F.normalize(qc, p=2.0, dim=-1, eps=module.eps)
            kc = F.normalize(kc, p=2.0, dim=-1, eps=module.eps)

        q = torch.cat([qx, qc], dim=2)
        k = torch.cat([kx, kc], dim=2)
        v = torch.cat([vx, vc], dim=2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        weights = scores.softmax(dim=-1)
        if dropout > 0.0 and module.training:
            weights = F.dropout(weights, p=dropout)
        attended = weights @ v

        att_x = attended[:, :, :tx, :]
        att_c = attended[:, :, tx:, :]

        out = module.x_out_proj(att_x.transpose(1, 2).reshape(b, tx, heads * head_dim))
        ctx_out = module.c_out_proj(att_c.transpose(1, 2).reshape(b, tc, heads * head_dim))
        return out, ctx_out

    return run


@operator(
    "joint_attention",
    identity="mmdit_attention",
    summary="Dual-stream MMDiT joint self-attention across image and context streams.",
    shape="x[B, T_x, D_x], context[B, T_c, D_c] -> out[B, T_x, D_x], ctx_out[B, T_c, D_c]",
    relation=_relation,
    args={
        "heads": Arg(int, min=1, help="Number of joint attention heads."),
        "head_dim": Arg(int, 0, min=0, positional=False, help="Per-head dimension; 0 defaults to D // heads."),
        "qk_norm": Arg(str, "none", choices=("none", "rms_norm", "l2_norm"), positional=False, help="QK normalization."),
        "bias": Arg(bool, True, positional=False, help="Whether linear projections have bias."),
        "dropout": Arg(float, 0.0, min=0.0, max=1.0, exclusive_max=True, positional=False, help="Dropout probability."),
        "eps": Arg(float, 1e-6, min=0.0, exclusive_min=True, positional=False, help="Epsilon for RMSNorm."),
    },
    examples=[
        Example(
            "c = linear(x, 32)\nx1, c1 = joint_attention(x, c, heads=4, head_dim=16)\nadd(x1, c1)",
            ("B", 8, 32),
            ("B", 8, 32),
            "MMDiT dual-stream joint attention block.",
        ),
    ],
    reference=_reference,
    category="sequence",
)
class JointAttention(nn.Module):
    """Dual-stream Joint Attention (MMDiT / Flux) operator."""

    def __init__(
        self,
        heads,
        head_dim=0,
        qk_norm="none",
        bias=True,
        dropout=0.0,
        eps=1e-6,
        *,
        D_x=0,
        D_c=0,
        D=0,
    ):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim) if head_dim > 0 else ((int(D_x) if D_x > 0 else int(D)) // self.heads)
        self.qk_norm = str(qk_norm)
        self.bias = bool(bias)
        self.dropout = float(dropout)
        self.eps = float(eps)

        d_x = int(D_x) if D_x > 0 else (int(D) if D > 0 else (self.heads * self.head_dim))
        d_c = int(D_c) if D_c > 0 else d_x
        self.d_x = d_x
        self.d_c = d_c

        inner_dim = self.heads * self.head_dim
        self.x_q_proj = nn.Linear(d_x, inner_dim, bias=self.bias)
        self.x_k_proj = nn.Linear(d_x, inner_dim, bias=self.bias)
        self.x_v_proj = nn.Linear(d_x, inner_dim, bias=self.bias)
        self.x_out_proj = nn.Linear(inner_dim, d_x, bias=self.bias)

        self.c_q_proj = nn.Linear(d_c, inner_dim, bias=self.bias)
        self.c_k_proj = nn.Linear(d_c, inner_dim, bias=self.bias)
        self.c_v_proj = nn.Linear(d_c, inner_dim, bias=self.bias)
        self.c_out_proj = nn.Linear(inner_dim, d_c, bias=self.bias)

        if self.qk_norm == "rms_norm":
            self.x_q_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
            self.x_k_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
            self.c_q_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
            self.c_k_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

    def forward(self, x, context):
        b, tx, _ = x.shape
        _, tc, _ = context.shape

        qx = self.x_q_proj(x).view(b, tx, self.heads, self.head_dim).transpose(1, 2)
        kx = self.x_k_proj(x).view(b, tx, self.heads, self.head_dim).transpose(1, 2)
        vx = self.x_v_proj(x).view(b, tx, self.heads, self.head_dim).transpose(1, 2)

        qc = self.c_q_proj(context).view(b, tc, self.heads, self.head_dim).transpose(1, 2)
        kc = self.c_k_proj(context).view(b, tc, self.heads, self.head_dim).transpose(1, 2)
        vc = self.c_v_proj(context).view(b, tc, self.heads, self.head_dim).transpose(1, 2)

        if self.qk_norm == "rms_norm":
            qx = self.x_q_norm(qx)
            kx = self.x_k_norm(kx)
            qc = self.c_q_norm(qc)
            kc = self.c_k_norm(kc)
        elif self.qk_norm == "l2_norm":
            qx = F.normalize(qx, p=2.0, dim=-1, eps=self.eps)
            kx = F.normalize(kx, p=2.0, dim=-1, eps=self.eps)
            qc = F.normalize(qc, p=2.0, dim=-1, eps=self.eps)
            kc = F.normalize(kc, p=2.0, dim=-1, eps=self.eps)

        q = torch.cat([qx, qc], dim=2)
        k = torch.cat([kx, kc], dim=2)
        v = torch.cat([vx, vc], dim=2)

        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )

        att_x = attended[:, :, :tx, :]
        att_c = attended[:, :, tx:, :]

        out = self.x_out_proj(att_x.transpose(1, 2).reshape(b, tx, self.heads * self.head_dim))
        ctx_out = self.c_out_proj(att_c.transpose(1, 2).reshape(b, tc, self.heads * self.head_dim))
        return out, ctx_out

    def extra_repr(self):
        return (
            f"heads={self.heads}, head_dim={self.head_dim}, "
            f"qk_norm='{self.qk_norm}', dropout={self.dropout}"
        )
