# `broadcast_mul`

Elementwise product of equal-rank tensors, broadcasting size-1 axes.

**Category:** arithmetic · **Identity:** `broadcast_mul@1`

## Shape

```text
a, b -> out
```

Relation: `equal rank and batch; each non-batch axis is equal or size 1; output takes the larger extent`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `a` | input | `a` | compute |
| `b` | input | `b` | compute |
| `out` | output | `out` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Compute ``a * b`` with equal-rank, non-batch broadcasting.

For example, ``[B,T,D] * [B,1,D]`` applies one channel gate per image.
Both operands must have the same batch and rank; reshape style vectors
explicitly. Gradients sum over broadcast axes, including for higher
derivatives. No parameters or buffers; train and eval are identical.
Use ``mul`` when broadcasting should be forbidden.

## Examples

### Example 1

A per-image channel gate scales every token.

```python
h = linear(8)
g = mean(h, 1, keepdim=True)
broadcast_mul(h, g)
```

Input `['B', 6, 4]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 4] -> [B, 6, 8]  dtype=float32
index  name  operation      input shapes              output shapes
0      n0    linear         x=[B, 6, 4]               out=[B, 6, 8]
1      n1    mean           x=[B, 6, 8]               out=[B, 1, 8]
2      n2    broadcast_mul  a=[B, 6, 8], b=[B, 1, 8]  out=[B, 6, 8]
```

Parameters: 40
