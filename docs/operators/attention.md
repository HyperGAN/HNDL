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
| `rope` | bool | `False` | — | Apply rotary position embeddings to the queries and keys before attending. |

## Description

Multi-head self-attention on a ``[B, T, D]`` sequence: ``B`` examples of
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
