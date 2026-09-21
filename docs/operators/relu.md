# `relu`

Rectified linear unit, max(x, 0).

**Category:** activation · **Identity:** `relu@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Out-of-place ReLU; shared branches are never mutated.

## Examples

### Example 1

```python
linear(64)
relu()
linear()
```

Input `['B', 128]` → output `['B', 10]`.

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 128]    out=[B, 64]
1      n1    relu       x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 8,906
