# `sub`

Elementwise difference of two tensors with identical shapes.

**Category:** arithmetic · **Identity:** `sub@1`

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

``out = a - b`` with no broadcasting; the two operands must agree on
rank, every axis extent, and dtype, and the difference is computed in the
plan compute dtype. Order matters, so both tensor inputs must be supplied
explicitly: ``sub(h, saved)`` is ``h - saved``.

## Examples

### Example 1

A residual that subtracts the shortcut; both inputs are explicit.

```python
saved = x
linear(8, name="branch")
relu()
h = linear(4)
sub(h, saved)
```

Input `['B', 4]` → output `['B', 4]`.

```text
Network: [B, 4] -> [B, 4]  dtype=float32
index  name    operation  input shapes        output shapes
0      branch  linear     x=[B, 4]            out=[B, 8]
1      n1      relu       x=[B, 8]            out=[B, 8]
2      n2      linear     x=[B, 8]            out=[B, 4]
3      n3      sub        a=[B, 4], b=[B, 4]  out=[B, 4]
```

Parameters: 76

### Example 2

Sequences work too: the two halves of the feature axis are differenced.

```python
a, b = split(4, dim=2)
sub(a, b)
```

Input `['B', 6, 8]` → output `['B', 6, 4]`.

```text
Network: [B, 6, 8] -> [B, 6, 4]  dtype=float32
index  name  operation  input shapes              output shapes
0      n0    split      x=[B, 6, 8]               first=[B, 6, 4], rest=[B, 6, 4]
1      n1    sub        a=[B, 6, 4], b=[B, 6, 4]  out=[B, 6, 4]
```

Parameters: 0

### Example 3

Images: the operand shapes must match exactly.

```python
saved = x
conv(3, kernel_size=3, padding=1)
h = tanh()
sub(h, saved)
```

Input `['B', 3, 8, 8]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 3, 8, 8]  dtype=float32
index  name  operation  input shapes                    output shapes
0      n0    conv       x=[B, 3, 8, 8]                  out=[B, 3, 8, 8]
1      n1    tanh       x=[B, 3, 8, 8]                  out=[B, 3, 8, 8]
2      n2    sub        a=[B, 3, 8, 8], b=[B, 3, 8, 8]  out=[B, 3, 8, 8]
```

Parameters: 84
