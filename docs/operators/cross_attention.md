# `cross_attention`

Multi-head attention with queries from one sequence and keys and values from another.

**Category:** sequence · **Identity:** `cross_attention@1`

## Shape

```text
x[B, T, D], context[B, S, D_ctx] -> out[B, T, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `context` | input | `context[B, S, D_ctx]` | compute |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `heads` (positional) | int | required | >= 1 | Number of attention heads; must divide the query width D. |
| `bias` | bool | `True` | — | Add a learned bias to the four projections. |
| `dropout` | float | `0.0` | >= 0; < 1 | Dropout probability on the attention weights while training; 0 disables it. |

## Description

Attends from a query sequence `x` of shape `[B, T, D]` to a context
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

## Examples

### Example 1

The first 4 positions query the remaining 8; 2 heads of width 4.

```python
q, kv = split(4)
cross_attention(q, kv, 2)
```

Input `['B', 12, 8]` → output `['B', 4, 8]`.

```text
Network: [B, 12, 8] -> [B, 4, 8]  dtype=float32
index  name  operation        input shapes                    output shapes
0      n0    split            x=[B, 12, 8]                    first=[B, 4, 8], rest=[B, 8, 8]
1      n1    cross_attention  x=[B, 4, 8], context=[B, 8, 8]  out=[B, 4, 8]
```

Parameters: 288

### Example 2

The context is 16 wide while the queries stay 8 wide: D_ctx need not equal D.

```python
q, kv = split(4)
c = linear(kv, 16)
cross_attention(q, c, 2)
```

Input `['B', 12, 8]` → output `['B', 4, 8]`.

```text
Network: [B, 12, 8] -> [B, 4, 8]  dtype=float32
index  name  operation        input shapes                     output shapes
0      n0    split            x=[B, 12, 8]                     first=[B, 4, 8], rest=[B, 8, 8]
1      n1    linear           x=[B, 8, 8]                      out=[B, 8, 16]
2      n2    cross_attention  x=[B, 4, 8], context=[B, 8, 16]  out=[B, 4, 8]
```

Parameters: 560

### Example 3

Unbiased projections, and the width that follows attention is inferred.

```python
q, kv = split(6)
cross_attention(q, kv, 4, bias=False)
linear()
```

Input `['B', 10, 16]` → output `['B', 6, 4]`.

```text
Network: [B, 10, 16] -> [B, 6, 4]  dtype=float32
index  name  operation        input shapes                      output shapes
0      n0    split            x=[B, 10, 16]                     first=[B, 6, 16], rest=[B, 4, 16]
1      n1    cross_attention  x=[B, 6, 16], context=[B, 4, 16]  out=[B, 6, 16]
2      n2    linear           x=[B, 6, 16]                      out=[B, 6, 4]
```

Parameters: 1,092
