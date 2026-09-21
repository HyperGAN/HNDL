# `instance_norm`

Normalize every channel of every example over its own spatial positions.

**Category:** normalization · **Identity:** `instance_norm@1`

## Shape

```text
x[B, C, ...] -> out[B, C, ...]
```

Relation: `x and out share the shape; rank 3 [B, C, L] or rank 4 [B, C, H, W]`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, ...]` | compute |
| `out` | output | `out[B, C, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `eps` | float | `1e-05` | > 0 | Added to the variance before the square root. |
| `affine` | bool | `False` | — | Learn a per-channel scale and bias. |
| `num_features` | int | inferred | >= 1; <= 2147483647; binds `C` | Channels at axis 1. Inferred from the incoming tensor. |

## Description

Normalizes each channel of each example independently over its spatial
positions, with no interaction between examples:

```text
out[b, c] = weight[c] * (x[b, c] - mean(x[b, c])) / sqrt(var(x[b, c]) + eps) + bias[c]
```

Supported ranks are 3 `[B, C, L]` and 4 `[B, C, H, W]`; the rank is fixed
at build time from the resolved input shape, so the module is exactly
`nn.InstanceNorm1d` for rank 3 and `nn.InstanceNorm2d` for rank 4. A
rank-2 `[B, C]` input is rejected during resolution, because a single
value per channel has no variance to normalize.

There are no running statistics (`track_running_stats` is always false),
so train and eval mode behave identically and the layer is deterministic
per example. Parameters are `weight` and `bias` when `affine` is true, one
value per channel; with the default `affine=False` the layer has no
parameters and no buffers at all.

Statistics are computed in the plan's compute dtype. A constant channel
normalizes to zero, up to `eps`.

## Examples

### Example 1

Each of the 8 channels is normalized per example over height and width.

```python
conv(8, kernel_size=3, padding=1)
instance_norm()
relu()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation      input shapes    output shapes
0      n0    conv           x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    instance_norm  x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
2      n2    relu           x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 2

The style-transfer arrangement: normalize, then a learned per-channel affine.

```python
conv(16, kernel_size=3, padding=1)
instance_norm(affine=True)
tanh()
conv(3, kernel_size=3, padding=1)
```

Input `['B', 3, 16, 16]` → output `['B', 3, 16, 16]`.

```text
Network: [B, 3, 16, 16] -> [B, 3, 16, 16]  dtype=float32
index  name  operation      input shapes       output shapes
0      n0    conv           x=[B, 3, 16, 16]   out=[B, 16, 16, 16]
1      n1    instance_norm  x=[B, 16, 16, 16]  out=[B, 16, 16, 16]
2      n2    tanh           x=[B, 16, 16, 16]  out=[B, 16, 16, 16]
3      n3    conv           x=[B, 16, 16, 16]  out=[B, 3, 16, 16]
```

Parameters: 915

### Example 3

A [B, C, L] input normalizes each of the 4 channels over its 16 positions.

```python
instance_norm()
flatten()
linear()
```

Input `['B', 4, 16]` → output `['B', 10]`.

```text
Network: [B, 4, 16] -> [B, 10]  dtype=float32
index  name  operation      input shapes  output shapes
0      n0    instance_norm  x=[B, 4, 16]  out=[B, 4, 16]
1      n1    flatten        x=[B, 4, 16]  out=[B, 64]
2      n2    linear         x=[B, 64]     out=[B, 10]
```

Parameters: 650
