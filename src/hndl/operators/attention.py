import math

import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import PAIR, Arg, Example, operator
from ._equalized import EqualLinear

ROPE_BASE = 10000.0


def _validate(args):
    height, width = args["spatial_shape"]
    if args["relative_position_bias"]:
        if height < 1 or width < 1:
            raise HNDLError("E_ARGUMENT",
                            "relative_position_bias=True needs spatial_shape=(H, W) with positive H and W")
    elif height or width:
        raise HNDLError("E_ARGUMENT",
                        "spatial_shape describes the token grid of relative_position_bias=True; "
                        "set relative_position_bias=True or drop spatial_shape")


def _relation(s):
    heads = s.args["heads"]
    width = None
    positions = None
    for port in ("x", "out"):
        shape = s.shape(port)
        if shape is None or len(shape) != 3:
            continue
        if width is None and shape[2] is not None:
            width = shape[2]
        if positions is None and shape[1] is not None:
            positions = shape[1]
    if s.args.get("relative_position_bias"):
        grid_height, grid_width = s.args.get("spatial_shape") or (0, 0)
        if grid_height > 0 and grid_width > 0:
            tokens = grid_height * grid_width
            if positions is not None and positions != tokens:
                s.error("E_CONSTRAINT",
                        f"spatial_shape=({grid_height}, {grid_width}) covers {tokens} tokens, "
                        f"but the sequence has T={positions}")
            else:
                for port in ("x", "out"):
                    s.axis(port, 1, tokens)
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


def _relative_position_index(height, width):
    """The ``[T, T]`` table row of every (query, key) pair on a row-major ``H x W`` grid."""
    tokens = torch.arange(height * width)
    rows, columns = tokens // width, tokens % width
    row_offset = rows[:, None] - rows[None, :] + (height - 1)
    column_offset = columns[:, None] - columns[None, :] + (width - 1)
    return row_offset * (2 * width - 1) + column_offset


def _causal_bias(positions, device, dtype):
    blocked = torch.ones(positions, positions, device=device, dtype=torch.bool).triu(1)
    return torch.zeros(positions, positions, device=device, dtype=dtype).masked_fill(blocked, float("-inf"))


def _reference(module):
    """Explicit ``softmax(QK^T / sqrt(head_dim) + bias + mask) V`` without a fused kernel."""
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
        if module.relative_position_bias:
            scores = scores + module.position_bias()
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
    validate=_validate,
    args={
        "heads": Arg(int, min=1, help="Number of attention heads; must divide the model width D."),
        "causal": Arg(bool, False, positional=False,
                      help="Mask out future positions so each token attends only to itself and its past."),
        "dropout": Arg(float, 0.0, min=0, max=1, exclusive_max=True, positional=False,
                       help="Dropout probability on the attention weights, applied in training mode only."),
        "bias": Arg(bool, True, positional=False, help="Add a learned bias to each of the four projections."),
        "equalized": Arg(bool, False, positional=False,
                         help="Use runtime fan-in scaling and N(0,1) raw weights for all four linear projections."),
        "qkv_bias": Arg(bool, True, positional=False,
                        help="Narrow bias for the query, key and value projections; they carry a bias only "
                             "when both bias and qkv_bias are true."),
        "out_bias": Arg(bool, True, positional=False,
                        help="Narrow bias for the output projection; it carries a bias only when both bias "
                             "and out_bias are true."),
        "rope": Arg(bool, False, positional=False,
                    help="Apply rotary position embeddings to the queries and keys before attending."),
        "relative_position_bias": Arg(bool, False, positional=False,
                                      help="Learn a per-head 2D relative-position bias added to the attention "
                                           "logits; requires spatial_shape."),
        "spatial_shape": Arg(PAIR, 0, min=0, positional=False,
                             help="Token grid (H, W) behind the sequence, row-major, with H*W == T. Only "
                                  "meaningful with relative_position_bias=True; an int applies to both axes."),
    },
    reference=_reference,
    examples=[
        Example("attention(4)", ("B", 8, 32), ("B", 8, 32),
                "Bidirectional self-attention with 4 heads of width 8."),
        Example("attention(4, equalized=True)", ("B", 8, 32), ("B", 8, 32),
                "Equalized Q/K/V/output projections; softmax and head scaling stay unchanged."),
        Example("attention(2, causal=True)", ("B", 6, 16), ("B", 6, 16),
                "Causal masking makes the layer autoregressive."),
        Example("linear(16)\nattention(4, rope=True)\nlinear()", ("B", 10, 8), ("B", 10, 8),
                "The model width 16 is set by the projection; rotary embeddings need an even head dimension."),
        Example("attention(2, spatial_shape=(4, 4), relative_position_bias=True)", ("B", 16, 16), ("B", 16, 16),
                "The 16 tokens are read as a 4x4 grid and each head learns a bias per (dy, dx) offset."),
        Example("attention(4, spatial_shape=(8, 8), relative_position_bias=True, qkv_bias=False, out_bias=True)",
                ("B", 64, 32), ("B", 64, 32),
                "TransGAN's generator block: no bias on q/k/v, a bias on the output projection, and a "
                "relative-position bias over the 8x8 token grid."),
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
    A       = softmax(Q @ K.T / sqrt(head_dim) + bias + mask) # [B, heads, T, T]
    out     = (A @ V) -> [B, T, D] -> out @ Wo.T + bo
    ```

    ``heads`` must divide ``D``; the resolver reports ``E_CONSTRAINT`` when it
    does not. Submodules are ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj``,
    each an ``nn.Linear(D, D)``, so their parameters are ``q_proj.weight``,
    ``q_proj.bias`` and so on. The attention itself is computed by
    ``torch.nn.functional.scaled_dot_product_attention``, which picks a fused
    kernel when one is available.

    With ``equalized=True`` all four projections initialize raw weights N(0,1)
    and any biases at zero, applying ``weight / sqrt(D)`` at every forward.
    Gain and learning-rate multiplier are one. Relative-position tables and
    the attention-logit head scaling are unchanged. ``init=`` and checkpoints
    target raw projection weights, not their runtime-scaled values.

    **Biases.** ``bias`` is the one flag for all four projections. ``qkv_bias``
    and ``out_bias`` narrow it per projection: ``q_proj``, ``k_proj`` and
    ``v_proj`` carry a bias when ``bias and qkv_bias``, and ``o_proj`` when
    ``bias and out_bias``. Both narrow flags default to ``True``, so ``bias=``
    on its own behaves exactly as it always has, and
    ``attention(4, qkv_bias=False, out_bias=True)`` gives the unbiased
    projections with a biased output that ViT-style blocks use. A projection
    built without a bias has no ``bias`` parameter at all.

    **Masking.** With ``causal=True`` position ``t`` attends only to positions
    ``<= t``: the strictly upper triangle of the score matrix is set to
    ``-inf`` before the softmax. With ``causal=False`` every position attends
    to every other; there is no padding mask, so pad positions are attended to
    like any other position.

    **Dropout.** ``dropout`` is applied to the attention weights and only in
    training mode (``module.train()``); in eval mode the layer is
    deterministic. At ``dropout=0`` the forward pass draws no random numbers in
    either mode.

    **Relative position bias.** With ``relative_position_bias=True`` the
    sequence is read as a row-major ``H x W`` grid given by
    ``spatial_shape=(H, W)``, so token ``t`` sits at ``row = t // W``,
    ``col = t % W`` and ``H*W`` must equal ``T`` — the resolver reports
    ``E_CONSTRAINT`` when it does not, and infers ``T`` from the grid when the
    sequence length is still open. Each head then learns one bias per possible
    query-minus-key offset:

    ```text
    dy = row(q) - row(k) + (H - 1)            # 0 .. 2H-2
    dx = col(q) - col(k) + (W - 1)            # 0 .. 2W-2
    row_of_table = dy * (2W - 1) + dx         # 0 .. (2H-1)*(2W-1) - 1
    bias[head, q, k] = table[row_of_table, head]
    ```

    ``relative_position_bias_table`` is the learned parameter, of shape
    ``[(2H-1)*(2W-1), heads]``, and ``relative_position_index`` is the
    precomputed ``[T, T]`` gather, a non-persistent int64 buffer rebuilt from
    ``spatial_shape`` on every construction rather than stored in a checkpoint.
    The table is constructed at ``0.0``, so the layer starts as ordinary
    attention and learns the bias from there; construction overrides target it
    by name, as ``attention(4, spatial_shape=(8, 8), relative_position_bias=True,
    init={"relative_position_bias_table": 0.02})`` or
    ``trainable={"relative_position_bias_table": False}``.

    The resulting ``[heads, T, T]`` bias is added to the scaled logits before
    the softmax, as an additive ``attn_mask`` so the fused kernels still apply.
    With ``causal=True`` as well, the causal ``-inf`` mask and the bias are
    summed into one additive mask rather than handed to the kernel separately.
    The operator carries no windowing or shifting of its own: partition a
    feature map into windows with `chunk` and reassemble it with `concat`, and
    give this node the ``(H, W)`` of one window.

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
    ``rope`` and ``relative_position_bias`` are independent and compose.

    Everything else stays in the plan compute dtype.
    """

    def __init__(self, heads, causal, dropout, bias, rope, *, D,
                 qkv_bias=True, out_bias=True, relative_position_bias=False, spatial_shape=(0, 0),
                 equalized=False):
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
        self.relative_position_bias = bool(relative_position_bias)
        height, width = (spatial_shape, spatial_shape) if type(spatial_shape) is int else tuple(spatial_shape)
        self.spatial_shape = (int(height), int(width))
        linear = EqualLinear if equalized else nn.Linear
        self.q_proj = linear(D, D, bias=bias and qkv_bias)
        self.k_proj = linear(D, D, bias=bias and qkv_bias)
        self.v_proj = linear(D, D, bias=bias and qkv_bias)
        self.o_proj = linear(D, D, bias=bias and out_bias)
        if self.relative_position_bias:
            height, width = self.spatial_shape
            if height < 1 or width < 1:
                raise HNDLError("E_ARGUMENT",
                                "relative_position_bias=True needs spatial_shape=(H, W) with positive H and W, "
                                f"got {self.spatial_shape}")
            self.positions = height * width
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * height - 1) * (2 * width - 1), heads))
            self.register_buffer("relative_position_index", _relative_position_index(height, width),
                                 persistent=False)
        else:
            self.positions = None

    def position_bias(self):
        """The learned ``[heads, T, T]`` additive bias gathered from the table."""
        gathered = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        return gathered.view(self.positions, self.positions, self.heads).permute(2, 0, 1).contiguous()

    def forward(self, x):
        batch, positions, width = x.shape
        mask = None
        if self.relative_position_bias:
            if positions != self.positions:
                raise HNDLError("E_CONSTRAINT",
                                f"spatial_shape={self.spatial_shape} covers {self.positions} tokens, "
                                f"but the input carries T={positions}")
            mask = self.position_bias().to(dtype=x.dtype)
            if self.causal:
                mask = mask + _causal_bias(positions, x.device, mask.dtype)
        q = _split_heads(self.q_proj(x), self.heads, self.head_dim)
        k = _split_heads(self.k_proj(x), self.heads, self.head_dim)
        v = _split_heads(self.v_proj(x), self.heads, self.head_dim)
        if self.rope:
            cos, sin = _rope_tables(positions, self.head_dim, x.device, x.dtype)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and mask is None)
        merged = attended.transpose(1, 2).reshape(batch, positions, width)
        return self.o_proj(merged)

    def extra_repr(self):
        grid = f", spatial_shape={self.spatial_shape}" if self.relative_position_bias else ""
        return (f"heads={self.heads}, head_dim={self.head_dim}, causal={self.causal}, "
                f"dropout={self.dropout}, rope={self.rope}, "
                f"relative_position_bias={self.relative_position_bias}{grid}")
