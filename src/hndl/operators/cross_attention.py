import math

import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, operator
from .rms_norm import RMSNorm


def _relation(s):
    heads = s.args["heads"]
    kv_heads = s.args.get("kv_heads", 0)
    head_dim = s.args.get("head_dim", 0)
    if kv_heads > 0 and heads % kv_heads != 0:
        s.error("E_CONSTRAINT", f"heads={heads} must be divisible by kv_heads={kv_heads}")
    if head_dim == 0:
        for port in ("x", "out"):
            shape = s.shape(port)
            width = None if shape is None else shape[-1]
            if width is not None and width % heads:
                s.error("E_CONSTRAINT", f"heads={heads} must divide the query width D={width}")


def _reference(module):
    """Textbook attention: scores, softmax, weighted values, output projection."""
    def run(x, context):
        heads, head_dim = module.heads, module.head_dim
        kv_heads = getattr(module, "kv_heads", heads)
        batch, positions, _ = x.shape
        sources = context.shape[1]
        q = module.q_proj(x).view(batch, positions, heads, head_dim).transpose(1, 2)
        k = module.k_proj(context).view(batch, sources, kv_heads, head_dim).transpose(1, 2)
        v = module.v_proj(context).view(batch, sources, kv_heads, head_dim).transpose(1, 2)
        qk_norm = getattr(module, "qk_norm", "none")
        if qk_norm == "rms_norm":
            q, k = module.q_norm(q), module.k_norm(k)
        elif qk_norm == "l2_norm":
            eps = getattr(module, "eps", 1e-6)
            q = F.normalize(q, p=2, dim=-1, eps=eps)
            k = F.normalize(k, p=2, dim=-1, eps=eps)
        if kv_heads != heads:
            repeats = heads // kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        logit_softcap = getattr(module, "logit_softcap", 0.0)
        if logit_softcap > 0.0:
            scores = logit_softcap * torch.tanh(scores / logit_softcap)
        attended = scores.softmax(dim=-1) @ v
        return module.o_proj(attended.transpose(1, 2).reshape(batch, positions, heads * head_dim))
    return run


@operator(
    "cross_attention",
    summary="Multi-head attention with queries from one sequence and keys and values from another.",
    shape="x[B, T, D], context[B, S, D_ctx] -> out[B, T, D]",
    relation=_relation,
    args={
        "heads": Arg(int, min=1, help="Number of attention heads; must divide the query width D."),
        "bias": Arg(bool, True, positional=False, help="Add a learned bias to the four projections."),
        "dropout": Arg(float, 0.0, min=0, max=1, exclusive_max=True, positional=False,
                       help="Dropout probability on the attention weights while training; 0 disables it."),
        "qkv_bias": Arg(bool, True, positional=False, since="0.9.0",
                        help="Narrow bias for Q/K/V projections; carries bias when bias and qkv_bias are true."),
        "out_bias": Arg(bool, True, positional=False, since="0.9.0",
                        help="Narrow bias for output projection; carries bias when bias and out_bias are true."),
        "kv_heads": Arg(int, 0, min=0, positional=False, since="0.9.0",
                        help="Number of key/value heads for GQA/MQA; 0 means kv_heads == heads. Must divide heads."),
        "head_dim": Arg(int, 0, min=0, positional=False, since="0.9.0",
                        help="Per-head dimension override; 0 means D // heads."),
        "qk_norm": Arg(str, "none", choices=("none", "rms_norm", "l2_norm"), positional=False, since="0.9.0",
                       help="Per-head query/key normalization."),
        "logit_softcap": Arg(float, 0.0, min=0.0, positional=False, since="0.9.0",
                             help="Logit soft-capping value C; 0.0 means disabled."),
        "eps": Arg(float, 1e-6, min=0, exclusive_min=True, positional=False, since="0.9.0",
                   help="Epsilon for QK normalization."),
    },
    examples=[
        Example("q, kv = split(4)\ncross_attention(q, kv, 2)", ("B", 12, 8), ("B", 4, 8),
                "The first 4 positions query the remaining 8; 2 heads of width 4."),
        Example("q, kv = split(4)\nc = linear(kv, 16)\ncross_attention(q, c, 2)", ("B", 12, 8), ("B", 4, 8),
                "The context is 16 wide while the queries stay 8 wide: D_ctx need not equal D."),
        Example("q, kv = split(6)\ncross_attention(q, kv, 4, bias=False)\nlinear()", ("B", 10, 16), ("B", 6, 4),
                "Unbiased projections, and the width that follows attention is inferred."),
    ],
    reference=_reference,
    category="sequence",
)
class CrossAttention(nn.Module):
    """Attends from a query sequence `x` of shape `[B, T, D]` to a context
    sequence of shape `[B, S, D_ctx]`. Positions are axis 1 and features are
    the last axis; the two sequences may differ in both length and width, but
    share the batch.

    ```text
    q = x @ Wq.T + bq                      # [B, T, D]
    k = context @ Wk.T + bk                # [B, S, D]
    v = context @ Wv.T + bv                # [B, S, D]
    per head: a = softmax(q_h @ k_h.T / sqrt(D / heads)) @ v_h
    out = concat(a_0 .. a_{heads-1}) @ Wo.T + bo
    ```

    `heads` must divide `D`; each head works on `D / heads` features. There is
    no mask: every query position attends to every context position, which is
    the usual encoder-decoder cross-attention.

    Submodules are `q_proj` (`D -> D`), `k_proj` and `v_proj` (`D_ctx -> D`),
    and `o_proj` (`D -> D`), each an `nn.Linear` carrying `weight` and, when
    `bias` is true, `bias`.

    In train mode `dropout` is applied to the attention weights; in eval mode
    the layer is deterministic. The default `dropout=0` behaves identically in
    both modes. Activations stay in the plan compute dtype; PyTorch's attention
    kernel may accumulate the softmax in float32 for stability in float16 and
    bfloat16, so reduced-precision results can be slightly more accurate than
    the same arithmetic done by hand.
    """

    def __init__(self, heads, bias, dropout, *, D, D_ctx,
                 qkv_bias=True, out_bias=True, kv_heads=0, head_dim=0,
                 qk_norm="none", logit_softcap=0.0, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads > 0 else heads
        if self.heads % self.kv_heads != 0:
            raise HNDLError("E_CONSTRAINT", f"heads={heads} must be divisible by kv_heads={self.kv_heads}")
        if head_dim == 0:
            if D % heads:
                raise HNDLError("E_CONSTRAINT", f"heads={heads} must divide the query width D={D}")
            self.head_dim = D // heads
        else:
            self.head_dim = head_dim
        self.dropout = float(dropout)
        self.bias = bool(bias)
        self.qkv_bias = bool(qkv_bias)
        self.out_bias = bool(out_bias)
        self.qk_norm = qk_norm
        self.logit_softcap = float(logit_softcap)
        self.eps = float(eps)

        q_dim = self.heads * self.head_dim
        kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, q_dim, bias=self.bias and self.qkv_bias)
        self.k_proj = nn.Linear(D_ctx, kv_dim, bias=self.bias and self.qkv_bias)
        self.v_proj = nn.Linear(D_ctx, kv_dim, bias=self.bias and self.qkv_bias)
        self.o_proj = nn.Linear(q_dim, D, bias=self.bias and self.out_bias)

        if self.qk_norm == "rms_norm":
            self.q_norm = RMSNorm(self.eps, affine=True, D=self.head_dim)
            self.k_norm = RMSNorm(self.eps, affine=True, D=self.head_dim)

    def forward(self, x, context):
        batch, positions, _ = x.shape
        sources = context.shape[1]
        heads, kv_heads, head_dim = self.heads, self.kv_heads, self.head_dim
        q = self.q_proj(x).view(batch, positions, heads, head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch, sources, kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch, sources, kv_heads, head_dim).transpose(1, 2)

        if self.qk_norm == "rms_norm":
            q = self.q_norm(q)
            k = self.k_norm(k)
        elif self.qk_norm == "l2_norm":
            q = F.normalize(q, p=2, dim=-1, eps=self.eps)
            k = F.normalize(k, p=2, dim=-1, eps=self.eps)

        if self.logit_softcap > 0.0:
            if kv_heads != heads:
                repeats = heads // kv_heads
                k_rep = k.repeat_interleave(repeats, dim=1)
                v_rep = v.repeat_interleave(repeats, dim=1)
            else:
                k_rep = k
                v_rep = v
            scores = (q @ k_rep.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
            weights = scores.softmax(dim=-1)
            if self.dropout > 0.0 and self.training:
                weights = F.dropout(weights, p=self.dropout)
            attended = weights @ v_rep
        else:
            attended = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=(kv_heads != heads),
            )
        return self.o_proj(attended.transpose(1, 2).reshape(batch, positions, heads * head_dim))

    def extra_repr(self):
        opts = ""
        if self.kv_heads != self.heads:
            opts += f", kv_heads={self.kv_heads}"
        if not self.qkv_bias:
            opts += ", qkv_bias=False"
        if not self.out_bias:
            opts += ", out_bias=False"
        if self.qk_norm != "none":
            opts += f", qk_norm={self.qk_norm!r}"
        if self.logit_softcap > 0.0:
            opts += f", logit_softcap={self.logit_softcap}"
        return f"heads={self.heads}, head_dim={self.head_dim}, dropout={self.dropout}{opts}"
