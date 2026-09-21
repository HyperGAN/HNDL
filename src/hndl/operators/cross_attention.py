import math

from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, operator


def _relation(s):
    heads = s.args["heads"]
    for port in ("x", "out"):
        shape = s.shape(port)
        width = None if shape is None else shape[-1]
        if width is not None and width % heads:
            s.error("E_CONSTRAINT", f"heads={heads} must divide the query width D={width}")


def _reference(module):
    """Textbook attention: scores, softmax, weighted values, output projection."""
    def run(x, context):
        heads, head_dim = module.heads, module.head_dim
        batch, positions, width = x.shape
        sources = context.shape[1]
        q = module.q_proj(x).view(batch, positions, heads, head_dim).transpose(1, 2)
        k = module.k_proj(context).view(batch, sources, heads, head_dim).transpose(1, 2)
        v = module.v_proj(context).view(batch, sources, heads, head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attended = scores.softmax(dim=-1) @ v
        return module.o_proj(attended.transpose(1, 2).reshape(batch, positions, width))
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

    def __init__(self, heads, bias, dropout, *, D, D_ctx):
        super().__init__()
        if D % heads:
            raise HNDLError("E_CONSTRAINT", f"heads={heads} must divide the query width D={D}")
        self.heads = heads
        self.head_dim = D // heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(D, D, bias=bias)
        self.k_proj = nn.Linear(D_ctx, D, bias=bias)
        self.v_proj = nn.Linear(D_ctx, D, bias=bias)
        self.o_proj = nn.Linear(D, D, bias=bias)

    def forward(self, x, context):
        batch, positions, width = x.shape
        sources = context.shape[1]
        heads, head_dim = self.heads, self.head_dim
        q = self.q_proj(x).view(batch, positions, heads, head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch, sources, heads, head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch, sources, heads, head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, positions, width))

    def extra_repr(self):
        return f"heads={self.heads}, head_dim={self.head_dim}, dropout={self.dropout}"
