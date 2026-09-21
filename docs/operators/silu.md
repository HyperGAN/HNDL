# `silu`

Sigmoid linear unit (swish), x * sigmoid(x).

**Category:** activation · **Identity:** `silu@1`

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

Elementwise ``out = x * sigmoid(x)``, also known as swish.

Unlike `relu` the function is smooth everywhere and keeps a small negative
response, with a minimum of about ``-0.278`` near ``x = -1.278``; unlike
`sigmoid` it is unbounded above, so it does not saturate for large
positive inputs.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.

## Examples

### Example 1

A smooth alternative to relu() on [B, F] features.

```python
linear(64)
silu()
linear()
```

Input `['B', 128]` → output `['B', 10]`.

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 128]    out=[B, 64]
1      n1    silu       x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 8,906

### Example 2

On a [B, T, D] sequence the activation applies elementwise at every position.

```python
linear(32)
silu()
linear()
```

Input `['B', 6, 16]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 16] -> [B, 6, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 6, 16]  out=[B, 6, 32]
1      n1    silu       x=[B, 6, 32]  out=[B, 6, 32]
2      n2    linear     x=[B, 6, 32]  out=[B, 6, 8]
```

Parameters: 808

### Example 3

The norm-then-activation pairing used by diffusion U-Nets on [B, C, H, W] tensors.

```python
conv(8, kernel_size=3, padding=1)
group_norm(4)
silu()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation   input shapes    output shapes
0      n0    conv        x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    group_norm  x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
2      n2    silu        x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 240
