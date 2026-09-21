# `layer_norm`

Normalize the last axis of every position with a learned scale and bias.

**Category:** normalization · **Identity:** `layer_norm@1`

## Shape

```text
x[B, ..., D] -> out[B, ..., D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ..., D]` | compute |
| `out` | output | `out[B, ..., D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `eps` | float | `1e-05` | > 0 | Added to the variance before the square root; must be positive. |
| `affine` | bool | `True` | — | Learn a per-feature scale and bias of width D. When false the layer has no parameters. |

## Description

Normalizes each row of the last axis ``D`` to zero mean and unit
variance, then applies a learned per-feature affine:

```text
mean = mean(x, dim=-1)
var  = mean((x - mean)^2, dim=-1)            # biased: divided by D
out  = (x - mean) / sqrt(var + eps) * weight + bias
```

The statistics are taken over the last axis only, so a ``[B, D]`` input
normalizes each example and a ``[B, T, D]`` sequence normalizes each
position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
relation but would be normalized over ``W`` alone, which is almost never
what is wanted; reach for `group_norm` on images. The examples below
therefore cover only ranks 2 and 3.

Parameters are ``weight`` (initialized to ones) and ``bias``
(initialized to zeros), both of shape ``[D]``, and exist only when
``affine`` is true. There are no running statistics, so train and eval
behave identically. The computation runs in the plan's compute dtype.

## Examples

### Example 1

The usual placement: normalize the hidden width before the activation.

```python
linear(64)
layer_norm()
relu()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    linear      x=[B, 32]     out=[B, 64]
1      n1    layer_norm  x=[B, 64]     out=[B, 64]
2      n2    relu        x=[B, 64]     out=[B, 64]
3      n3    linear      x=[B, 64]     out=[B, 10]
```

Parameters: 2,890

### Example 2

On a [B, T, D] sequence each of the 12 positions is normalized independently.

```python
linear(64)
layer_norm()
linear()
```

Input `['B', 12, 32]` → output `['B', 12, 10]`.

```text
Network: [B, 12, 32] -> [B, 12, 10]  dtype=float32
index  name  operation   input shapes   output shapes
0      n0    linear      x=[B, 12, 32]  out=[B, 12, 64]
1      n1    layer_norm  x=[B, 12, 64]  out=[B, 12, 64]
2      n2    linear      x=[B, 12, 64]  out=[B, 12, 10]
```

Parameters: 2,890

### Example 3

A parameter-free normalization of the incoming features.

```python
layer_norm(eps=0.001, affine=False)
linear()
```

Input `['B', 16]` → output `['B', 4]`.

```text
Network: [B, 16] -> [B, 4]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    layer_norm  x=[B, 16]     out=[B, 16]
1      n1    linear      x=[B, 16]     out=[B, 4]
```

Parameters: 68
