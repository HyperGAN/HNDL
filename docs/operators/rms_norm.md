# `rms_norm`

Scale the last axis by its root-mean-square, with a learned per-feature gain.

**Category:** normalization · **Identity:** `rms_norm@1`

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
| `eps` | float | `1e-06` | > 0 | Added to the mean square before the reciprocal square root; must be positive. |
| `affine` | bool | `True` | — | Learn a per-feature gain of width D. When false the layer has no parameters. |

## Description

Root-mean-square normalization of the last axis ``D``: the cheaper
half of `layer_norm` that rescales without re-centering.

```text
out = x / sqrt(mean(x^2, dim=-1) + eps) * weight
```

The statistics are taken over the last axis only, so a ``[B, D]`` input
normalizes each example and a ``[B, T, D]`` sequence normalizes each
position independently. Rank-4 ``[B, C, H, W]`` tensors satisfy the shape
relation but would be normalized over ``W`` alone, which is almost never
what is wanted; reach for `group_norm` on images. The examples below
therefore cover only ranks 2 and 3.

The mean square and its reciprocal square root are accumulated in
``float32`` even when the plan's compute dtype is ``float16`` or
``bfloat16``, because squaring a half-precision activation overflows
around 256; the normalized tensor is cast back to the input dtype before
the gain is applied, so the layer's inputs, outputs and parameters stay
in the plan dtype.

The single parameter is ``weight`` of shape ``[D]``, initialized to ones
and present only when ``affine`` is true. There are no running
statistics, so train and eval behave identically.

## Examples

### Example 1

Pre-activation normalization without the mean subtraction of `layer_norm`.

```python
linear(64)
rms_norm()
relu()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    rms_norm   x=[B, 64]     out=[B, 64]
2      n2    relu       x=[B, 64]     out=[B, 64]
3      n3    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,826

### Example 2

On a [B, T, D] sequence each of the 12 positions is normalized independently.

```python
linear(64)
rms_norm()
linear()
```

Input `['B', 12, 32]` → output `['B', 12, 10]`.

```text
Network: [B, 12, 32] -> [B, 12, 10]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    linear     x=[B, 12, 32]  out=[B, 12, 64]
1      n1    rms_norm   x=[B, 12, 64]  out=[B, 12, 64]
2      n2    linear     x=[B, 12, 64]  out=[B, 12, 10]
```

Parameters: 2,826

### Example 3

A parameter-free variant, useful directly on the network input.

```python
rms_norm(eps=0.0001, affine=False)
linear()
```

Input `['B', 16]` → output `['B', 4]`.

```text
Network: [B, 16] -> [B, 4]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    rms_norm   x=[B, 16]     out=[B, 16]
1      n1    linear     x=[B, 16]     out=[B, 4]
```

Parameters: 68
