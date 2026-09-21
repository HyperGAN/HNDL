# `transformer_block`

Pre-norm transformer block: residual self-attention followed by a residual feed-forward.

**Category:** sequence · **Identity:** `transformer_block@1`

## Shape

```text
x[B, T, D] -> out[B, T, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `heads` (positional) | int | required | >= 1 | Number of attention heads; must divide the model width D. |
| `mlp_ratio` | int | `4` | >= 1 | Inner width of the feed-forward as a multiple of D; hidden = mlp_ratio * D. |
| `activation` | str | `"gelu"` | one of `gelu`, `gelu_tanh`, `relu`, `silu`, `quick_gelu` | Nonlinearity inside the feed-forward branch. |
| `norm` | str | `"layer_norm"` | one of `layer_norm`, `rms_norm` | Normalization in front of each branch: "layer_norm" or "rms_norm". |
| `causal` | bool | `False` | — | Mask out future positions so each token attends only to itself and its past. |
| `dropout` | float | `0.0` | >= 0; < 1 | Dropout probability on the attention weights and after the feed-forward activation. |
| `layer_scale` | bool | `False` | — | Scale each residual branch by a learned per-feature gain of width D. |
| `layer_scale_init` | float | `1e-05` | >= 0 | Value the layer-scale gains start at; ignored when layer_scale is false. |
| `eps` | float | `1e-05` | > 0 | Epsilon of both normalizations; must be positive. |

## Description

One pre-norm transformer block on a ``[B, T, D]`` sequence: ``B``
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

## Examples

### Example 1

A ViT-style encoder block: 4 heads of width 8 and a feed-forward of width 128.

```python
transformer_block(4)
```

Input `['B', 8, 32]` → output `['B', 8, 32]`.

```text
Network: [B, 8, 32] -> [B, 8, 32]  dtype=float32
index  name  operation          input shapes  output shapes
0      n0    transformer_block  x=[B, 8, 32]  out=[B, 8, 32]
```

Parameters: 12,704

### Example 2

Two stacked causal blocks with RMS normalization, the modern decoder recipe.

```python
transformer_block(2, causal=True, norm="rms_norm")
transformer_block(2, causal=True, norm="rms_norm")
```

Input `['B', 6, 16]` → output `['B', 6, 16]`.

```text
Network: [B, 6, 16] -> [B, 6, 16]  dtype=float32
index  name  operation          input shapes  output shapes
0      n0    transformer_block  x=[B, 6, 16]  out=[B, 6, 16]
1      n1    transformer_block  x=[B, 6, 16]  out=[B, 6, 16]
```

Parameters: 6,496

### Example 3

The model width 16 comes from the projection; layer scale starts the block near the identity.

```python
linear(16)
transformer_block(2, mlp_ratio=2, layer_scale=True)
linear()
```

Input `['B', 6, 8]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 8] -> [B, 6, 8]  dtype=float32
index  name  operation          input shapes  output shapes
0      n0    linear             x=[B, 6, 8]   out=[B, 6, 16]
1      n1    transformer_block  x=[B, 6, 16]  out=[B, 6, 16]
2      n2    linear             x=[B, 6, 16]  out=[B, 6, 8]
```

Parameters: 2,536
