# `identity`

Pass the tensor through unchanged, as a named node.

**Category:** activation · **Identity:** `identity@1`

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

Returns `x` unchanged: `out = x`, the same tensor object, with no copy,
no parameters, and no effect on the autograd graph beyond passing the
gradient straight through.

It exists so a configuration can name a point in the graph — `h =
identity(name="hidden")` gives a branch or a later `add` something to refer
to — and so a slot in a chain can be filled without changing the
computation, for instance when a normalization or activation is being
ablated. Shape, dtype, and device are preserved exactly for any supported
rank: rank 2 `[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`.
Because the shape relation is the identity, it is fully transparent to
resolution in both directions. Behavior is identical in train and eval
mode.

## Examples

### Example 1

A named tap: the passthrough gives a residual add something to refer back to.

```python
h = identity(name="hidden")
linear(8)
relu()
y = linear(4)
add(y, h)
```

Input `['B', 4]` → output `['B', 4]`.

```text
Network: [B, 4] -> [B, 4]  dtype=float32
index  name    operation  input shapes        output shapes
0      hidden  identity   x=[B, 4]            out=[B, 4]
1      n1      linear     x=[B, 4]            out=[B, 8]
2      n2      relu       x=[B, 8]            out=[B, 8]
3      n3      linear     x=[B, 8]            out=[B, 4]
4      n4      add        a=[B, 4], b=[B, 4]  out=[B, 4]
```

Parameters: 76

### Example 2

A no-op placeholder that keeps the shape contract flowing through the chain.

```python
linear(64)
identity()
relu()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    identity   x=[B, 64]     out=[B, 64]
2      n2    relu       x=[B, 64]     out=[B, 64]
3      n3    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,762

### Example 3

Any supported rank passes through unchanged.

```python
conv(8, kernel_size=3, padding=1)
identity()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    identity   x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224
