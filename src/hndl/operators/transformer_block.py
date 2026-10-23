import torch
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, operator
from .attention import Attention, _reference as _attention_reference
from .feed_forward import ACTIVATIONS, FeedForward, _reference as _feed_forward_reference
from .layer_norm import LayerNorm, _reference as _layer_norm_reference
from .rms_norm import RMSNorm, _reference as _rms_norm_reference
from .swiglu import SwiGLU, _reference as _swiglu_reference

NORMS = ("layer_norm", "rms_norm")


def _validate(args):
    if not args.get("rope") and args.get("rope_theta") != 10000.0:
        raise HNDLError("E_ARGUMENT", "rope_theta is only valid when rope=True")


def _relation(s):
    """``heads`` must divide the model width ``D``, whichever side fixes it."""
    heads = s.args["heads"]
    kv_heads = s.args.get("kv_heads", 0)
    if kv_heads > 0 and heads % kv_heads != 0:
        s.error("E_CONSTRAINT", f"heads={heads} must be divisible by kv_heads={kv_heads}")
    for port in ("x", "out"):
        shape = s.shape(port)
        if shape is not None and len(shape) == 3 and shape[2] is not None:
            width = shape[2]
            if width % heads:
                s.error("E_CONSTRAINT", f"heads={heads} must divide the model width D={width}")
            if s.args.get("rope") and (width // heads) % 2:
                s.error("E_CONSTRAINT", f"rope needs an even head dimension, but D={width} over heads={heads} gives {width // heads}")
            return


def _reference(module):
    """A functional rebuild of the block delegating to its submodules' reference implementations."""
    norm_ref = _rms_norm_reference if module.norm_kind == "rms_norm" else _layer_norm_reference
    norm1_ref = norm_ref(module.norm1)
    attn_ref = _attention_reference(module.attn)
    norm2_ref = norm_ref(module.norm2)
    if getattr(module, "ffn_kind", "mlp") == "swiglu":
        ffn_ref = _swiglu_reference(module.ffn)
    else:
        ffn_ref = _feed_forward_reference(module.ffn)

    post_norm = getattr(module, "post_norm", False)

    def run(x):
        if post_norm:
            branch = attn_ref(x)
            x = norm1_ref(x + (branch if module.gamma1 is None else branch * module.gamma1))
            branch = ffn_ref(x)
            return norm2_ref(x + (branch if module.gamma2 is None else branch * module.gamma2))
        branch = attn_ref(norm1_ref(x))
        x = x + (branch if module.gamma1 is None else branch * module.gamma1)
        branch = ffn_ref(norm2_ref(x))
        return x + (branch if module.gamma2 is None else branch * module.gamma2)

    return run


@operator(
    "transformer_block",
    summary="Pre-norm transformer block: residual self-attention followed by a residual feed-forward.",
    shape="x[B, T, D] -> out[B, T, D]",
    relation=_relation,
    validate=_validate,
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
        "ffn_kind": Arg(str, "mlp", choices=("mlp", "swiglu"), positional=False, since="0.9.0",
                        help='Feed-forward architecture: "mlp" or "swiglu".'),
        "hidden": Arg(int, 0, min=0, positional=False, since="0.9.0",
                      help="Explicit inner width of the feed-forward; 0 means mlp_ratio * D."),
        "rope": Arg(bool, False, positional=False, since="0.9.0",
                    help="Enable rotary position embeddings in self-attention."),
        "rope_theta": Arg(float, 10000.0, min=1.0, exclusive_min=True, positional=False, since="0.9.0",
                          help="Base frequency for rotary embeddings."),
        "kv_heads": Arg(int, 0, min=0, positional=False, since="0.9.0",
                        help="Number of key/value heads for GQA in self-attention; 0 means heads."),
        "qk_norm": Arg(str, "none", choices=("none", "rms_norm", "l2_norm"), positional=False, since="0.9.0",
                       help="Query/key normalization in self-attention."),
        "bias": Arg(bool, True, positional=False, since="0.9.0",
                    help="Add learned biases to projections and normalizations."),
        "head_dim": Arg(int, 0, min=0, positional=False, since="0.10.0",
                        help="Attention per-head dimension; 0 defaults to D // heads."),
        "window_size": Arg(int, 0, min=0, positional=False, since="0.10.0",
                           help="Sliding-window attention window size; 0 means full context."),
        "logit_softcap": Arg(float, 0.0, min=0.0, positional=False, since="0.10.0",
                             help="Attention logit soft-capping value C; 0.0 means disabled."),
        "qkv_bias": Arg(bool, True, positional=False, since="0.10.0",
                        help="Learn separate biases on attention Q/K/V projections."),
        "out_bias": Arg(bool, True, positional=False, since="0.10.0",
                        help="Learn bias on attention output projection."),
        "zero_centered": Arg(bool, False, positional=False, since="0.10.0",
                             help="Initialize RMSNorm weight to 0.0 and evaluate as (1 + weight) * rms_norm(x)."),
        "post_norm": Arg(bool, False, positional=False, since="0.10.0",
                         help="Post-LN transformer architecture instead of pre-LN."),
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

    (LLaMA-style architectures can use `ffn_kind="swiglu"` and `rope=True`
    directly within `transformer_block`, or compose `attention(rope=True)` and
    `swiglu` by hand.)

    **Train and eval.** ``dropout`` is passed to both branches — the attention
    weights and the feed-forward hidden activations — and is active in
    training mode only. With ``dropout=0`` the block is deterministic in both
    modes; there are no running statistics either way. Everything is computed
    in the plan's compute dtype, apart from the float32 accumulation inside
    ``rms_norm``.
    """

    def __init__(self, heads, mlp_ratio, activation, norm, causal, dropout,
                 layer_scale, layer_scale_init, eps, *, D,
                 ffn_kind="mlp", hidden=0, rope=False, rope_theta=10000.0,
                 kv_heads=0, qk_norm="none", bias=True,
                 head_dim=0, window_size=0, logit_softcap=0.0,
                 qkv_bias=True, out_bias=True, zero_centered=False, post_norm=False):
        super().__init__()
        self.norm_kind = norm
        self.ffn_kind = ffn_kind
        self.bias = bool(bias)
        self.zero_centered = bool(zero_centered)
        self.post_norm = bool(post_norm)
        if norm == "rms_norm":
            self.norm1 = RMSNorm(eps=eps, affine=True, D=D, zero_centered=zero_centered)
            self.norm2 = RMSNorm(eps=eps, affine=True, D=D, zero_centered=zero_centered)
        else:
            self.norm1 = LayerNorm(eps=eps, affine=True, bias=self.bias, D=D)
            self.norm2 = LayerNorm(eps=eps, affine=True, bias=self.bias, D=D)

        self.attn = Attention(
            heads=heads, causal=causal, dropout=dropout, bias=self.bias, rope=rope, D=D,
            rope_theta=rope_theta, kv_heads=kv_heads, qk_norm=qk_norm, eps=eps,
            head_dim=head_dim, window_size=window_size, logit_softcap=logit_softcap,
            qkv_bias=qkv_bias, out_bias=out_bias,
        )

        actual_hidden = hidden if hidden > 0 else (mlp_ratio * int(D))
        if ffn_kind == "swiglu":
            self.ffn = SwiGLU(hidden=actual_hidden, bias=self.bias, D=D)
        else:
            self.ffn = FeedForward(
                hidden=actual_hidden, activation=activation, dropout=dropout, bias=self.bias, D=D,
            )

        if layer_scale:
            self.gamma1 = nn.Parameter(torch.full((int(D),), float(layer_scale_init)))
            self.gamma2 = nn.Parameter(torch.full((int(D),), float(layer_scale_init)))
        else:
            self.register_parameter("gamma1", None)
            self.register_parameter("gamma2", None)

    def forward(self, x):
        if self.post_norm:
            branch = self.attn(x)
            x = self.norm1(x + branch if self.gamma1 is None else torch.addcmul(x, branch, self.gamma1))
            branch = self.ffn(x)
            return self.norm2(x + branch if self.gamma2 is None else torch.addcmul(x, branch, self.gamma2))
        branch = self.attn(self.norm1(x))
        x = x + branch if self.gamma1 is None else torch.addcmul(x, branch, self.gamma1)
        branch = self.ffn(self.norm2(x))
        return x + branch if self.gamma2 is None else torch.addcmul(x, branch, self.gamma2)

    def extra_repr(self):
        opts = f", ffn_kind={self.ffn_kind!r}" if self.ffn_kind != "mlp" else ""
        if not self.bias:
            opts += ", bias=False"
        if getattr(self, "zero_centered", False):
            opts += ", zero_centered=True"
        if getattr(self, "post_norm", False):
            opts += ", post_norm=True"
        return f"norm={self.norm_kind!r}, layer_scale={self.gamma1 is not None}{opts}"
