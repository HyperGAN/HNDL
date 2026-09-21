# `quick_gelu`

CLIP's fast GELU approximation, x * sigmoid(1.702 * x).

**Category:** activation · **Identity:** `quick_gelu@1`

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

Elementwise ``out = x * sigmoid(1.702 * x)``.

This is the "quick GELU" used by OpenAI's CLIP and by the models derived
from it. The logistic curve approximates the Gaussian cumulative
distribution function, so the result tracks `gelu` within about 1e-2 in
absolute value while costing one sigmoid instead of an erf. A model
trained with this activation must keep it: it is close to, but not
numerically interchangeable with, `gelu`.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters and behaves
identically in train and eval mode. It is computed out of place in the
input dtype; the 1.702 coefficient is applied as a plain multiply, so
float16 and bfloat16 activations are never silently upcast.

## Examples

### Example 1

A drop-in replacement for gelu() on [B, F] features.

```python
linear(64)
quick_gelu()
linear()
```

Input `['B', 128]` → output `['B', 10]`.

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    linear      x=[B, 128]    out=[B, 64]
1      n1    quick_gelu  x=[B, 64]     out=[B, 64]
2      n2    linear      x=[B, 64]     out=[B, 10]
```

Parameters: 8,906

### Example 2

On a [B, T, D] sequence the activation applies elementwise at every position.

```python
linear(32)
quick_gelu()
linear()
```

Input `['B', 6, 16]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 16] -> [B, 6, 8]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    linear      x=[B, 6, 16]  out=[B, 6, 32]
1      n1    quick_gelu  x=[B, 6, 32]  out=[B, 6, 32]
2      n2    linear      x=[B, 6, 32]  out=[B, 6, 8]
```

Parameters: 808

### Example 3

Elementwise over every channel and spatial position of a [B, C, H, W] tensor.

```python
conv(8, kernel_size=3, padding=1)
quick_gelu()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation   input shapes    output shapes
0      n0    conv        x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    quick_gelu  x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224
