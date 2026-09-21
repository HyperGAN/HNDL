# `group_norm`

Normalize channel groups per example, with learned per-channel affine.

**Category:** normalization · **Identity:** `group_norm@1`

## Shape

```text
x[B, C, ...] -> out[B, C, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, ...]` | compute |
| `out` | output | `out[B, C, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `num_groups` (positional) | int | required | >= 1 | Number of channel groups; must divide the channel count. |
| `num_channels` | int | inferred | >= 1; <= 2147483647; binds `C` | Channels at axis 1. Inferred from the incoming tensor. |
| `eps` | float | `1e-05` | > 0 | Added to the variance for stability. |
| `affine` | bool | `True` | — | Learn per-channel scale and bias. |

## Description

Splits the channel axis into ``num_groups`` groups and normalizes each
group over its channels and spatial positions using population
statistics. Behavior is identical in train and eval mode. Parameters are
``weight`` and ``bias`` when ``affine`` is true.

## Examples

### Example 1

```python
conv(16, kernel_size=3, padding=1)
group_norm(4)
relu()
```

Input `['B', 3, 8, 8]` → output `['B', 16, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 16, 8, 8]  dtype=float32
index  name  operation   input shapes     output shapes
0      n0    conv        x=[B, 3, 8, 8]   out=[B, 16, 8, 8]
1      n1    group_norm  x=[B, 16, 8, 8]  out=[B, 16, 8, 8]
2      n2    relu        x=[B, 16, 8, 8]  out=[B, 16, 8, 8]
```

Parameters: 480
