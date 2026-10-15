# `mul`

Elementwise product of two tensors with identical shapes.

**Category:** arithmetic · **Identity:** `mul@1`

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

``out = a * b`` (the Hadamard product) with no broadcasting; the two
operands must agree on rank, every axis extent, and dtype, and the product
is computed in the plan compute dtype. Both tensor inputs must be supplied
explicitly. Use ``scale`` instead to multiply by a constant.

## Examples

### Example 1

A gate: a second branch scales the first elementwise.

```python
h = linear(4)
g = linear(x, 4)
g = tanh(g)
mul(h, g)
```

Input `['B', 8]` → output `['B', 4]`.

```text
Network: [B, 8] -> [B, 4]  dtype=float32
name  operation  input shapes                                output shapes
n0    linear     x=input:x:[B, 8]                            out=[B, 4]
n1    linear     x=input:x:[B, 8]                            out=[B, 4]
n2    tanh       x=node:n1/out:[B, 4]                        out=[B, 4]
n3    mul        a=node:n0/out:[B, 4], b=node:n2/out:[B, 4]  out=[B, 4]
```

Parameters: 72

### Example 2

Sequences work too: the two halves of the feature axis are multiplied.

```python
a, b = split(4, dim=2)
mul(a, b)
```

Input `['B', 6, 8]` → output `['B', 6, 4]`.

```text
Network: [B, 6, 8] -> [B, 6, 4]  dtype=float32
name  operation  input shapes                                         output shapes
n0    split      x=input:x:[B, 6, 8]                                  first=[B, 6, 4], rest=[B, 6, 4]
n1    mul        a=node:n0/first:[B, 6, 4], b=node:n0/rest:[B, 6, 4]  out=[B, 6, 4]
```

Parameters: 0

### Example 3

Images: the operand shapes must match exactly.

```python
mask = x
conv(3, kernel_size=3, padding=1)
h = tanh()
mul(h, mask)
```

Input `['B', 3, 8, 8]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 3, 8, 8]  dtype=float32
name  operation  input shapes                                        output shapes
n0    conv       x=input:x:[B, 3, 8, 8]                              out=[B, 3, 8, 8]
n1    tanh       x=node:n0/out:[B, 3, 8, 8]                          out=[B, 3, 8, 8]
n2    mul        a=node:n1/out:[B, 3, 8, 8], b=input:x:[B, 3, 8, 8]  out=[B, 3, 8, 8]
```

Parameters: 84
