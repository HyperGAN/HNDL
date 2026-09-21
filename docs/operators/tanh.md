# `tanh`

Hyperbolic tangent, squashing values into (-1, 1).

**Category:** activation · **Identity:** `tanh@1`

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

Elementwise ``tanh(x)``.

## Examples

### Example 1

A common final activation for image generators.

```python
conv(3, kernel_size=3, padding=1)
tanh()
```

Input `['B', 8, 16, 16]` → output `['B', 3, 16, 16]`.

```text
Network: [B, 8, 16, 16] -> [B, 3, 16, 16]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    conv       x=[B, 8, 16, 16]  out=[B, 3, 16, 16]
1      n1    tanh       x=[B, 3, 16, 16]  out=[B, 3, 16, 16]
```

Parameters: 219
