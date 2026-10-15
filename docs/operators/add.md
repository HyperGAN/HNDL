# `add`

Elementwise sum of two tensors with identical shapes.

**Category:** arithmetic · **Identity:** `add@1`

## Shape

```text
a[B, ...], b[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `a` | input | `a[B, ...]` | compute |
| `b` | input | `b[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

``out = a + b`` with no broadcasting; shapes, dtype, and batch must
match exactly. Both tensor inputs must be supplied explicitly.

## Examples

### Example 1

A residual connection: both inputs are explicit.

```python
saved = x
linear(8, name="branch")
relu()
h = linear(4)
add(h, saved)
```

Input `['B', 4]` → output `['B', 4]`.

```text
Network: [B, 4] -> [B, 4]  dtype=float32
name    operation  input shapes                            output shapes
branch  linear     x=input:x:[B, 4]                        out=[B, 8]
n1      relu       x=node:branch/out:[B, 8]                out=[B, 8]
n2      linear     x=node:n1/out:[B, 8]                    out=[B, 4]
n3      add        a=node:n2/out:[B, 4], b=input:x:[B, 4]  out=[B, 4]
```

Parameters: 76
