# `hardswish`

Piecewise-linear approximation of swish, x * relu6(x + 3) / 6.

**Category:** activation · **Identity:** `hardswish@1`

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

Elementwise piecewise-linear approximation of `swish = x * sigmoid(x)`:

```
out = 0                    if x <= -3
out = x                    if x >= +3
out = x * (x + 3) / 6      otherwise
```

Equivalently `out = x * relu6(x + 3) / 6`. The hard sigmoid gate needs
only clamping and a multiply, which makes this noticeably cheaper than
`swish` on hardware without a fast `exp`, while tracking it closely. The
function is continuous, its derivative is discontinuous at `x = -3` and
`x = +3`, and negative inputs below -3 are zeroed exactly.

The operation is computed out of place in the activation dtype
(float32, float16, or bfloat16) and preserves any supported shape — rank 2
`[B, F]`, rank 3 `[B, T, D]`, or rank 4 `[B, C, H, W]`. It has no
parameters and behaves identically in train and eval mode.

## Examples

### Example 1

The activation used by mobile-scale image backbones.

```python
conv(8, kernel_size=3, padding=1)
hardswish()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    hardswish  x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 2

A cheap smooth-ish alternative to relu() in a classifier head.

```python
linear(64)
hardswish()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    hardswish  x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,762

### Example 3

Elementwise on a [B, T, D] sequence.

```python
linear(16)
hardswish()
```

Input `['B', 4, 8]` → output `['B', 4, 16]`.

```text
Network: [B, 4, 8] -> [B, 4, 16]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 8]   out=[B, 4, 16]
1      n1    hardswish  x=[B, 4, 16]  out=[B, 4, 16]
```

Parameters: 144
