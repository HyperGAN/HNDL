import math

import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, operator

ROPE_BASE = 10000.0


def _relation(s):
    heads = s.args["heads"]
    width = None
    for port in ("x", "out"):
        shape = s.shape(port)
        if shape is not None and len(shape) == 3 and shape[2] is not None:
            width = shape[2]
            break
    if width is None:
        return
    if width % heads:
        s.error("E_CONSTRAINT", f"heads={heads} must divide the model width D={width}")
    elif s.args.get("rope") and (width // heads) % 2:
        s.error("E_CONSTRAINT",
                f"rope needs an even head dimension, but D={width} over heads={heads} gives {width // heads}")


def _split_heads(tensor, heads, head_dim):
    batch, positions, _ = tensor.shape
    return tensor.view(batch, positions, heads, head_dim).transpose(1, 2)


def _rotate_half(tensor):
    half = tensor.shape[-1] // 2
    left, right = tensor[..., :half], tensor[..., half:]
    return torch.cat((-right, left), dim=-1)


def _rope_tables(positions, head_dim, device, dtype):
    """``cos``/``sin`` of shape ``[1, 1, T, head_dim]`` for the rotate-half convention."""
    # The angles are built in float32 even for a float16 plan: 1 / base**(i/d)
    # underflows and the position product loses resolution in half precision.
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inverse_frequency = torch.pow(torch.tensor(ROPE_BASE, device=device, dtype=torch.float32), -exponent)
    angles = torch.arange(positions, device=device, dtype=torch.float32)[:, None] * inverse_frequency[None, :]
    duplicated = torch.cat((angles, angles), dim=-1)
    return duplicated.cos().to(dtype)[None, None], duplicated.sin().to(dtype)[None, None]


def _apply_rope(tensor, cos, sin):
    return tensor * cos + _rotate_half(tensor) * sin


def _reference(module):
    """Explicit ``softmax(QK^T / sqrt(head_dim) + mask) V`` without a fused kernel."""
    def run(x):
        batch, positions, width = x.shape
        heads, head_dim = module.heads, module.head_dim
        q = _split_heads(module.q_proj(x), heads, head_dim)
        k = _split_heads(module.k_proj(x), heads, head_dim)
        v = _split_heads(module.v_proj(x), heads, head_dim)
        if module.rope:
            cos, sin = _rope_tables(positions, head_dim, x.device, x.dtype)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        if module.causal:
            blocked = torch.ones(positions, positions, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(blocked, float("-inf"))
        weights = scores.softmax(dim=-1)
        merged = (weights @ v).transpose(1, 2).reshape(batch, positions, width)
        return module.o_proj(merged)
    return run


@operator(
    "attention",
    summary="Multi-head self-attention over a [B, T, D] sequence.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    args={
        "heads": Arg(int, min=1, help="Number of attention heads; must divide the model width D."),
        "causal": Arg(bool, False, positional=False,
                      help="Mask out future positions so each token attends only to itself and its past."),
        "dropout": Arg(float, 0.0, min=0, max=1, exclusive_max=True, positional=False,
                       help="Dropout probability on the attention weights, applied in training mode only."),
        "bias": Arg(bool, True, positional=False, help="Add a learned bias to each of the four projections."),
        "rope": Arg(bool, False, positional=False,
                    help="Apply rotary position embeddings to the queries and keys before attending."),
    },
    reference=_reference,
    examples=[
        Example("attention(4)", ("B", 8, 32), ("B", 8, 32),
                "Bidirectional self-attention with 4 heads of width 8."),
        Example("attention(2, causal=True)", ("B", 6, 16), ("B", 6, 16),
                "Causal masking makes the layer autoregressive."),
        Example("linear(16)\nattention(4, rope=True)\nlinear()", ("B", 10, 8), ("B", 10, 8),
                "The model width 16 is set by the projection; rotary embeddings need an even head dimension."),
    ],
    category="sequence",
)
class Attention(nn.Module):
    """Multi-head self-attention on a ``[B, T, D]`` sequence: ``B`` examples of
    ``T`` positions with ``D`` features each. Queries, keys and values all come
    from the same tensor.

    ```text
    Q, K, V = x @ Wq.T + bq, x @ Wk.T + bk, x @ Wv.T + bv    # each [B, T, D]
    Q, K, V -> [B, heads, T, head_dim]                       # head_dim = D / heads
    A       = softmax(Q @ K.T / sqrt(head_dim) + mask)       # [B, heads, T, T]
    out     = (A @ V) -> [B, T, D] -> out @ Wo.T + bo
    ```

    ``heads`` must divide ``D``; the resolver reports ``E_CONSTRAINT`` when it
    does not. Submodules are ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj``,
    each an ``nn.Linear(D, D, bias=bias)``, so their parameters are
    ``q_proj.weight``, ``q_proj.bias`` and so on. The attention itself is
    computed by ``torch.nn.functional.scaled_dot_product_attention``, which
    picks a fused kernel when one is available.

    **Masking.** With ``causal=True`` position ``t`` attends only to positions
    ``<= t``: the strictly upper triangle of the score matrix is set to
    ``-inf`` before the softmax. With ``causal=False`` every position attends
    to every other; there is no padding mask, so pad positions are attended to
    like any other position.

    **Dropout.** ``dropout`` is applied to the attention weights and only in
    training mode (``module.train()``); in eval mode the layer is
    deterministic.

    **Rotary embeddings.** With ``rope=True`` each query and key head is
    rotated by its absolute position before the scores are formed, which makes
    the scores depend on relative distance. The rotate-half convention over the
    full head dimension is used, with base 10000:

    ```text
    theta_i   = 10000 ** (-2i / head_dim)     # i = 0 .. head_dim/2 - 1
    angle     = t * theta_i                   # at position t
    rotate_half([a, b]) = [-b, a]             # a, b the two halves of the head
    q_rotated = q * cos(angle) + rotate_half(q) * sin(angle)
    ```

    The tables are recomputed from ``T`` on every forward pass, so there are no
    position buffers and no maximum sequence length. The angles are built in
    float32 and cast to the compute dtype, because ``10000 ** (-2i/head_dim)``
    underflows in float16. ``rope`` therefore needs an even ``head_dim``. At
    ``T == 1`` the only angle is zero, so the layer matches ``rope=False``.

    Everything else stays in the plan compute dtype.
    """

    def __init__(self, heads, causal, dropout, bias, rope, *, D):
        super().__init__()
        if D % heads:
            raise HNDLError("E_CONSTRAINT", f"heads={heads} must divide the model width D={D}")
        self.heads = heads
        self.head_dim = D // heads
        if rope and self.head_dim % 2:
            raise HNDLError("E_CONSTRAINT", f"rope needs an even head dimension, got {self.head_dim}")
        self.causal = bool(causal)
        self.dropout = float(dropout)
        self.rope = bool(rope)
        self.q_proj = nn.Linear(D, D, bias=bias)
        self.k_proj = nn.Linear(D, D, bias=bias)
        self.v_proj = nn.Linear(D, D, bias=bias)
        self.o_proj = nn.Linear(D, D, bias=bias)

    def forward(self, x):
        batch, positions, width = x.shape
        q = _split_heads(self.q_proj(x), self.heads, self.head_dim)
        k = _split_heads(self.k_proj(x), self.heads, self.head_dim)
        v = _split_heads(self.v_proj(x), self.heads, self.head_dim)
        if self.rope:
            cos, sin = _rope_tables(positions, self.head_dim, x.device, x.dtype)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        merged = attended.transpose(1, 2).reshape(batch, positions, width)
        return self.o_proj(merged)

    def extra_repr(self):
        return (f"heads={self.heads}, head_dim={self.head_dim}, causal={self.causal}, "
                f"dropout={self.dropout}, rope={self.rope}")
