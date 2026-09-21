# `elu`

Exponential linear unit: identity above zero, saturating below.

**Category:** activation · **Identity:** `elu@1`

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
| `alpha` (positional) | float | `1.0` | — | Negative saturation value; the activation approaches -alpha as x decreases. |

## Description

Elementwise exponential linear unit:

```
out = x                          if x > 0
out = alpha * (exp(x) - 1)       if x <= 0
```

The function is continuous at the origin with value 0, and for negative
inputs it saturates smoothly at `-alpha` instead of clamping to zero, so
unlike ReLU it keeps a nonzero gradient there. `alpha = 1` gives the
standard form with a continuous derivative at 0.

The operation is computed out of place in the activation dtype
(float32, float16, or bfloat16), preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]` — has no
parameters, and behaves identically in train and eval mode.

## Examples

### Example 1

A hidden activation with nonzero gradient for negative inputs.

```python
linear(64)
elu()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    elu        x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,762

### Example 2

Half the negative saturation on a [B, C, H, W] image.

```python
conv(8, kernel_size=3, padding=1)
elu(0.5)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    elu        x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 3

Elementwise on a [B, T, D] sequence.

```python
linear(16)
elu()
```

Input `['B', 4, 8]` → output `['B', 4, 16]`.

```text
Network: [B, 4, 8] -> [B, 4, 16]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 8]   out=[B, 4, 16]
1      n1    elu        x=[B, 4, 16]  out=[B, 4, 16]
```

Parameters: 144
