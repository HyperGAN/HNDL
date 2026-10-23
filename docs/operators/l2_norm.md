# `l2_norm`

Parameter-free L2 normalization along a specified axis.

**Category:** normalization · **Identity:** `l2_norm@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

Relation: `out has the shape of x; dim must resolve to a non-batch axis`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dim` (positional) | int | `-1` | — | Axis along which L2 normalization is computed. |
| `eps` | float | `1e-06` | > 0 | Small positive constant to avoid division by zero. |

## Description

Normalize vectors along a specified dimension to unit L2 norm.

Applies parameter-free L2 normalization:
``out = x / max(norm(x, 2, dim), eps)``.

## Examples

### Example 1

Normalizes hidden vectors along the feature dimension.

```python
linear(64)
l2_norm()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shape  output shape
0      n0    linear     [B, 32]      [B, 64]
1      n1    l2_norm    [B, 64]      [B, 64]
2      n2    linear     [B, 64]      [B, 10]
```

Parameters: 2,762
