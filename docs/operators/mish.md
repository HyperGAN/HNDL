# `mish`

Self-gated smooth activation, x * tanh(softplus(x)).

**Category:** activation · **Identity:** `mish@1`

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

Elementwise self-gated activation:

```
out = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
```

The gate `tanh(softplus(x))` rises smoothly from 0 to 1, so `mish`
approaches the identity for large positive inputs and decays towards 0 for
large negative ones, with a small negative dip near `x = -1`. It is smooth
everywhere (unlike ReLU) and unbounded above, which keeps gradients alive
on the negative side.

`softplus` inside the gate uses PyTorch's stable formulation, so the
activation is safe in float16 and bfloat16 as well as float32; it is
computed in the activation dtype without upcasting. It is elementwise, so
it preserves any supported shape — rank 2 `[B, F]`, rank 3 `[B, T, D]`, or
rank 4 `[B, C, H, W]` — has no parameters, and behaves identically in
train and eval mode.

## Examples

### Example 1

A smooth drop-in replacement for relu().

```python
linear(64)
mish()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    mish       x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,762

### Example 2

Elementwise on a [B, C, H, W] image.

```python
conv(8, kernel_size=3, padding=1)
mish()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    mish       x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 3

Elementwise on a [B, T, D] sequence.

```python
linear(16)
mish()
```

Input `['B', 4, 8]` → output `['B', 4, 16]`.

```text
Network: [B, 4, 8] -> [B, 4, 16]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 8]   out=[B, 4, 16]
1      n1    mish       x=[B, 4, 16]  out=[B, 4, 16]
```

Parameters: 144
