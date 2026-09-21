# `batch_norm`

Normalize each channel over the batch and spatial axes, tracking running statistics.

**Category:** normalization · **Identity:** `batch_norm@1`

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
| `eps` (positional) | float | `1e-05` | > 0 | Added to the variance before the square root. |
| `momentum` | float | `0.1` | >= 0; <= 1 | Weight of the current batch in the running-statistics update. |
| `affine` | bool | `True` | — | Learn a per-channel scale and bias. |
| `track_running_stats` | bool | `True` | — | Keep running_mean/running_var buffers and use them in eval mode. |
| `num_features` | int | inferred | >= 1; <= 2147483647; binds `C` | Channels at axis 1. Inferred from the incoming tensor. |

## Description

Normalizes each channel of axis 1 across the batch and every remaining
axis:

```text
out = weight * (x - mean) / sqrt(var + eps) + bias
```

Supported ranks are 2 `[B, C]`, 3 `[B, C, L]` and 4 `[B, C, H, W]`; the
rank is fixed at build time from the resolved input shape, so the module
is exactly `nn.BatchNorm1d` for ranks 2 and 3 and `nn.BatchNorm2d` for
rank 4, including its state-dict layout.

In train mode the statistics come from the current batch, and the
`running_mean` / `running_var` buffers are updated once per forward as
`running = (1 - momentum) * running + momentum * batch` (the variance uses
the unbiased batch estimate, the normalization the biased one);
`num_batches_tracked` counts the updates. In eval mode the buffers are
used instead, so the layer becomes a fixed affine map. With
`track_running_stats=False` there are no buffers and batch statistics are
used in both modes.

Parameters are `weight` and `bias` when `affine` is true, one value per
channel. Statistics are accumulated in the plan's compute dtype.

## Examples

### Example 1

Normalizes the 16 convolution channels over batch, height and width.

```python
conv(16, kernel_size=3, padding=1)
batch_norm()
relu()
conv(3, kernel_size=3, padding=1)
```

Input `['B', 3, 8, 8]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 3, 8, 8]  dtype=float32
index  name  operation   input shapes     output shapes
0      n0    conv        x=[B, 3, 8, 8]   out=[B, 16, 8, 8]
1      n1    batch_norm  x=[B, 16, 8, 8]  out=[B, 16, 8, 8]
2      n2    relu        x=[B, 16, 8, 8]  out=[B, 16, 8, 8]
3      n3    conv        x=[B, 16, 8, 8]  out=[B, 3, 8, 8]
```

Parameters: 915

### Example 2

On a [B, C] tensor every feature is a channel.

```python
linear(64)
batch_norm()
relu()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    linear      x=[B, 32]     out=[B, 64]
1      n1    batch_norm  x=[B, 64]     out=[B, 64]
2      n2    relu        x=[B, 64]     out=[B, 64]
3      n3    linear      x=[B, 64]     out=[B, 10]
```

Parameters: 2,890

### Example 3

A [B, C, L] input normalizes each of the 4 channels over the batch and the 16 positions.

```python
batch_norm()
flatten()
linear()
```

Input `['B', 4, 16]` → output `['B', 10]`.

```text
Network: [B, 4, 16] -> [B, 10]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    batch_norm  x=[B, 4, 16]  out=[B, 4, 16]
1      n1    flatten     x=[B, 4, 16]  out=[B, 64]
2      n2    linear      x=[B, 64]     out=[B, 10]
```

Parameters: 658

### Example 4

Stating the channel count backward fixes the preceding projection at 64.

```python
linear()
batch_norm(num_features=64, affine=False)
relu()
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation   input shapes  output shapes
0      n0    linear      x=[B, 32]     out=[B, 64]
1      n1    batch_norm  x=[B, 64]     out=[B, 64]
2      n2    relu        x=[B, 64]     out=[B, 64]
3      n3    linear      x=[B, 64]     out=[B, 10]
```

Parameters: 2,762
