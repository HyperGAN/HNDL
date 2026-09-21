# `softplus`

Smooth positive activation, log(1 + exp(beta*x)) / beta.

**Category:** activation · **Identity:** `softplus@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `beta` (positional) | float | `1.0` | > 0 | Sharpness of the bend at zero; larger values approach ReLU. |
| `threshold` | float | `20.0` | > 0 | Above beta*x = threshold the function is evaluated as the identity for stability. |

## Description

Elementwise smooth approximation of ReLU:

```
out = log(1 + exp(beta * x)) / beta
```

The output is strictly positive and the function is differentiable
everywhere; its derivative is the logistic sigmoid `sigmoid(beta * x)`.
Larger `beta` sharpens the bend at the origin and the limit is ReLU.

For numerical stability the linear branch `out = x` is used wherever
`beta * x > threshold`, which is exact in floating point well before the
default threshold of 20. The rule applies unchanged in float16 and
bfloat16, where the identity branch avoids overflowing `exp`.

The operator is elementwise, so it preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — and it has no
parameters and no train/eval difference.

## Examples

### Example 1

A strictly positive hidden activation.

```python
linear(64)
softplus()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    softplus   x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,762

### Example 2

A sharper bend, closer to ReLU, on a [B, C, H, W] image.

```python
conv(8, kernel_size=3, padding=1)
softplus(2.0)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    softplus   x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 3

Elementwise on a [B, T, D] sequence.

```python
linear(16)
softplus(1.0, threshold=10.0)
```

Input `['B', 4, 8]` → output `['B', 4, 16]`.

```text
Network: [B, 4, 8] -> [B, 4, 16]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 8]   out=[B, 4, 16]
1      n1    softplus   x=[B, 4, 16]  out=[B, 4, 16]
```

Parameters: 144
