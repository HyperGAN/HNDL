# `where`

Ternary conditional selection: out = where(cond, a, b).

**Category:** core · **Identity:** `where@1`

## Shape

```text
cond[B, ...]:bool, a[B, ...], b[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `cond` | input | `cond[B, ...]:bool` | bool |
| `a` | input | `a[B, ...]` | compute |
| `b` | input | `b[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Ternary element-wise selection: out = torch.where(cond, a, b).

## Examples

### Example 1

Selects elements from x where mask is True, and b where False.

```python
b = linear(8)
mask = cast(x, 'bool')
out = where(mask, x, b)
```

Input `['B', 8]` → output `['B', 8]`.

```text
Network: [B, 8] -> [B, 8]  dtype=float32
name  operation  input shapes                                                     output shapes
n0    linear     x=input:x:[B, 8]                                                 out=[B, 8]
n1    cast       x=input:x:[B, 8]                                                 out=[B, 8]
n2    where      cond=node:n1/out:[B, 8], a=input:x:[B, 8], b=node:n0/out:[B, 8]  out=[B, 8]
```

Parameters: 72
