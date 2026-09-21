import math

import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator
from .attention import Attention
from .feed_forward import ACTIVATIONS, FeedForward
from .layer_norm import LayerNorm
from .rms_norm import RMSNorm

NORMS = ("layer_norm", "rms_norm")


def _relation(s):
    """``heads`` must divide the model width ``D``, whichever side fixes it."""
    heads = s.args["heads"]
    for port in ("x", "out"):
        shape = s.shape(port)
        if shape is not None and len(shape) == 3 and shape[2] is not None:
            if shape[2] % heads:
                s.error("E_CONSTRAINT", f"heads={heads} must divide the model width D={shape[2]}")
            return


def _normalize(norm, x):
    """The functional form of whichever normalization the block was built with."""
    if isinstance(norm, RMSNorm):
        return F.rms_norm(x, norm.normalized_shape, norm.weight, norm.eps)
    return F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)


def _attention(attn, x):
    """``softmax(QK^T / sqrt(head_dim) + mask) V`` written out, with no fused kernel."""
    batch, positions, width = x.shape
    heads, head_dim = attn.heads, attn.head_dim

    def split(projection):
        return F.linear(x, projection.weight, projection.bias).view(
            batch, positions, heads, head_dim).transpose(1, 2)

    q, k, v = split(attn.q_proj), split(attn.k_proj), split(attn.v_proj)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    if attn.causal:
        blocked = torch.ones(positions, positions, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(blocked, float("-inf"))
    merged = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, positions, width)
    return F.linear(merged, attn.o_proj.weight, attn.o_proj.bias)


def _feed_forward(ffn, x):
    hidden = ACTIVATIONS[ffn.activation](F.linear(x, ffn.up.weight, ffn.up.bias))
    if ffn.dropout.p:
        hidden = F.dropout(hidden, ffn.dropout.p, ffn.training)
    return F.linear(hidden, ffn.down.weight, ffn.down.bias)


def _reference(module):
    """A functional rebuild of the block from its own parameters, bypassing the submodules."""
    def run(x):
        branch = _attention(module.attn, _normalize(module.norm1, x))
        x = x + (branch if module.gamma1 is None else branch * module.gamma1)
        branch = _feed_forward(module.ffn, _normalize(module.norm2, x))
        return x + (branch if module.gamma2 is None else branch * module.gamma2)
    return run


@operator(
    "transformer_block",
    summary="Pre-norm transformer block: residual self-attention followed by a residual feed-forward.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    reference=_reference,
    args={
        "heads": Arg(int, min=1, help="Number of attention heads; must divide the model width D."),
        "mlp_ratio": Arg(int, 4, min=1, positional=False,
                         help="Inner width of the feed-forward as a multiple of D; hidden = mlp_ratio * D."),
        "activation": Arg(str, "gelu", positional=False, choices=tuple(ACTIVATIONS),
                          help="Nonlinearity inside the feed-forward branch."),
        "norm": Arg(str, "layer_norm", positional=False, choices=NORMS,
                    help='Normalization in front of each branch: "layer_norm" or "rms_norm".'),
        "causal": Arg(bool, False, positional=False,
                      help="Mask out future positions so each token attends only to itself and its past."),
        "dropout": Arg(float, 0.0, min=0, max=1, exclusive_max=True, positional=False,
                       help="Dropout probability on the attention weights and after the feed-forward activation."),
        "layer_scale": Arg(bool, False, positional=False,
                           help="Scale each residual branch by a learned per-feature gain of width D."),
        "layer_scale_init": Arg(float, 1e-5, min=0, positional=False,
                                help="Value the layer-scale gains start at; ignored when layer_scale is false."),
        "eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False,
                   help="Epsilon of both normalizations; must be positive."),
    },
    examples=[
        Example("transformer_block(4)", ("B", 8, 32), ("B", 8, 32),
                "A ViT-style encoder block: 4 heads of width 8 and a feed-forward of width 128."),
        Example('transformer_block(2, causal=True, norm="rms_norm")\n'
                'transformer_block(2, causal=True, norm="rms_norm")', ("B", 6, 16), ("B", 6, 16),
                "Two stacked causal blocks with RMS normalization, the modern decoder recipe."),
        Example("linear(16)\ntransformer_block(2, mlp_ratio=2, layer_scale=True)\nlinear()",
                ("B", 6, 8), ("B", 6, 8),
                "The model width 16 comes from the projection; layer scale starts the block near the identity."),
    ],
    category="sequence",
)
class TransformerBlock(nn.Module):
    """One pre-norm transformer block on a ``[B, T, D]`` sequence: ``B``
    examples of ``T`` positions with ``D`` features each. Both branches are
    residual and both normalize their input rather than their output:

    ```text
    x   = x + ls1 * attn(norm1(x))      # multi-head self-attention
    out = x + ls2 * ffn(norm2(x))       # position-wise feed-forward
    ```

    ``ls1`` and ``ls2`` are the layer-scale gains, and are absent (a plain sum)
    unless ``layer_scale`` is set. The block preserves its shape exactly, so
    blocks stack by repetition and ``D`` flows through untouched in both
    directions of the resolver.

    **Submodules and parameter names.** The four children are ``norm1``,
    ``attn``, ``norm2`` and ``ffn``, so the state dict holds

    | Path | Shape |
    | --- | --- |
    | `norm1.weight`, `norm1.bias` | `[D]` (no `bias` with `rms_norm`) |
    | `attn.{q,k,v,o}_proj.weight` | `[D, D]` |
    | `attn.{q,k,v,o}_proj.bias` | `[D]` |
    | `norm2.weight`, `norm2.bias` | `[D]` (no `bias` with `rms_norm`) |
    | `ffn.up.weight`, `ffn.up.bias` | `[hidden, D]`, `[hidden]` |
    | `ffn.down.weight`, `ffn.down.bias` | `[D, hidden]`, `[D]` |
    | `gamma1`, `gamma2` | `[D]`, only when `layer_scale` |

    with ``hidden = mlp_ratio * D``. Those paths are what ``init`` and
    ``trainable`` target, for example ``init={"gamma1": 0.1}``.

    **Attention.** `attention` with all four projections biased and no rotary
    embeddings. ``heads`` must divide ``D``; the resolver reports
    ``E_CONSTRAINT`` when it does not. With ``causal=True`` position ``t``
    attends only to positions ``<= t``, which makes a stack of blocks
    autoregressive; with ``causal=False`` every position sees every other and
    the block is a bidirectional encoder. There is no padding mask.

    **Feed-forward.** `feed_forward` of inner width ``mlp_ratio * D``, so the
    default ``mlp_ratio=4`` gives the classic ``4 * D`` transformer MLP.
    ``activation`` selects ``gelu``, ``gelu_tanh``, ``relu``, ``silu`` or
    ``quick_gelu``.

    **Normalization.** ``norm="layer_norm"`` re-centers and rescales each
    position; ``norm="rms_norm"`` only rescales and has no bias. Both use
    ``eps`` and both are affine. ``rms_norm`` accumulates its mean square in
    float32 before casting back, as that operator documents.

    **Layer scale.** With ``layer_scale=True`` each branch is multiplied by a
    learned vector of width ``D`` — ``gamma1`` for attention and ``gamma2``
    for the feed-forward — both filled with ``layer_scale_init``. A small
    initial value (the default ``1e-5``) starts the block at almost exactly
    the identity, which is what makes very deep stacks trainable.

    **Correspondences.**

    | Recipe | Arguments |
    | --- | --- |
    | GPT-2 | `transformer_block(heads, activation="gelu_tanh", causal=True)` |
    | ViT / BERT-style encoder | `transformer_block(heads)` |
    | LLaMA-style decoder | `transformer_block(heads, norm="rms_norm", causal=True)` |
    | CaiT / DINOv2 | `transformer_block(heads, layer_scale=True)` |

    (LLaMA additionally uses a gated `swiglu` feed-forward and rotary
    embeddings, which this block does not; compose `attention(rope=True)` and
    `swiglu` by hand for that.)

    **Train and eval.** ``dropout`` is passed to both branches — the attention
    weights and the feed-forward hidden activations — and is active in
    training mode only. With ``dropout=0`` the block is deterministic in both
    modes; there are no running statistics either way. Everything is computed
    in the plan's compute dtype, apart from the float32 accumulation inside
    ``rms_norm``.
    """

    def __init__(self, heads, mlp_ratio, activation, norm, causal, dropout,
                 layer_scale, layer_scale_init, eps, *, D):
        super().__init__()
        self.norm_kind = norm
        make_norm = RMSNorm if norm == "rms_norm" else LayerNorm
        self.norm1 = make_norm(eps=eps, affine=True, D=D)
        self.attn = Attention(heads=heads, causal=causal, dropout=dropout, bias=True, rope=False, D=D)
        self.norm2 = make_norm(eps=eps, affine=True, D=D)
        self.ffn = FeedForward(hidden=mlp_ratio * int(D), activation=activation, dropout=dropout, bias=True, D=D)
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.full((int(D),), float(layer_scale_init)))
            self.gamma2 = nn.Parameter(torch.full((int(D),), float(layer_scale_init)))
        else:
            self.register_parameter("gamma1", None)
            self.register_parameter("gamma2", None)

    def forward(self, x):
        branch = self.attn(self.norm1(x))
        x = x + (branch if self.gamma1 is None else branch * self.gamma1)
        branch = self.ffn(self.norm2(x))
        return x + (branch if self.gamma2 is None else branch * self.gamma2)

    def extra_repr(self):
        return f"norm={self.norm_kind!r}, layer_scale={self.gamma1 is not None}"
