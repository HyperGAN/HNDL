# `attention`

Multi-head self-attention over a [B, T, D] sequence.

**Category:** sequence · **Identity:** `attention@1`

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
| `causal` | bool | `False` | — | Mask out future positions so each token attends only to itself and its past. |
| `dropout` | float | `0.0` | >= 0; < 1 | Dropout probability on the attention weights, applied in training mode only. |
| `bias` | bool | `True` | — | Add a learned bias to each of the four projections. |
| `qkv_bias` | bool | `True` | — | Narrow bias for the query, key and value projections; they carry a bias only when both bias and qkv_bias are true. |
| `out_bias` | bool | `True` | — | Narrow bias for the output projection; it carries a bias only when both bias and out_bias are true. |
| `rope` | bool | `False` | — | Apply rotary position embeddings to the queries and keys before attending. |
| `relative_position_bias` | bool | `False` | — | Learn a per-head 2D relative-position bias added to the attention logits; requires spatial_shape. |
| `spatial_shape` | pair | `(0, 0)` | >= 0 | Token grid (H, W) behind the sequence, row-major, with H*W == T. Only meaningful with relative_position_bias=True; an int applies to both axes. |

## Description

Multi-head self-attention on a ``[B, T, D]`` sequence: ``B`` examples of
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

## Examples

### Example 1

Bidirectional self-attention with 4 heads of width 8.

```python
attention(4)
```

Input `['B', 8, 32]` → output `['B', 8, 32]`.

```text
Network: [B, 8, 32] -> [B, 8, 32]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    attention  x=[B, 8, 32]  out=[B, 8, 32]
```

Parameters: 4,224

### Example 2

Causal masking makes the layer autoregressive.

```python
attention(2, causal=True)
```

Input `['B', 6, 16]` → output `['B', 6, 16]`.

```text
Network: [B, 6, 16] -> [B, 6, 16]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    attention  x=[B, 6, 16]  out=[B, 6, 16]
```

Parameters: 1,088

### Example 3

The model width 16 is set by the projection; rotary embeddings need an even head dimension.

```python
linear(16)
attention(4, rope=True)
linear()
```

Input `['B', 10, 8]` → output `['B', 10, 8]`.

```text
Network: [B, 10, 8] -> [B, 10, 8]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    linear     x=[B, 10, 8]   out=[B, 10, 16]
1      n1    attention  x=[B, 10, 16]  out=[B, 10, 16]
2      n2    linear     x=[B, 10, 16]  out=[B, 10, 8]
```

Parameters: 1,368

### Example 4

The 16 tokens are read as a 4x4 grid and each head learns a bias per (dy, dx) offset.

```python
attention(2, spatial_shape=(4, 4), relative_position_bias=True)
```

Input `['B', 16, 16]` → output `['B', 16, 16]`.

```text
Network: [B, 16, 16] -> [B, 16, 16]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    attention  x=[B, 16, 16]  out=[B, 16, 16]
```

Parameters: 1,186

### Example 5

TransGAN's generator block: no bias on q/k/v, a bias on the output projection, and a relative-position bias over the 8x8 token grid.

```python
attention(4, spatial_shape=(8, 8), relative_position_bias=True, qkv_bias=False, out_bias=True)
```

Input `['B', 64, 32]` → output `['B', 64, 32]`.

```text
Network: [B, 64, 32] -> [B, 64, 32]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    attention  x=[B, 64, 32]  out=[B, 64, 32]
```

Parameters: 5,028
