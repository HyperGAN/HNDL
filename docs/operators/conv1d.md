# `conv1d`

One-dimensional convolution over [B, C, L] or [B, T, C] signals.

**Category:** convolution · **Identity:** `conv1d@1`

## Shape

```text
x -> out
```

Relation: `x[B, C_in, L_in] -> out[B, C_out, L_out] (or x[B, L_in, C_in] -> out[B, L_out, C_out] when channels_last=True)`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `out_channels` (positional) | int | inferred | >= 1; <= 2147483647 | Output channels. Omit to infer from the consumer. |
| `in_channels` | int | inferred | >= 1; <= 2147483647 | Input channels. Normally inferred from the incoming tensor. |
| `kernel_size` | int | required | >= 1 | Kernel width in positions along the length axis. |
| `stride` | int | `1` | >= 1 | Step between kernel applications along the length axis. |
| `padding` | int | `0` | >= 0 | Zero padding added to each end of the length axis. |
| `dilation` | int | `1` | >= 1 | Spacing between kernel taps along the length axis. |
| `groups` | int | `1` | >= 1 | Channel groups; must divide input and output channels. |
| `bias` | bool | `True` | — | Add a learned per-channel bias. |
| `spectral_norm` | bool | `False` | — | Divide the weight by its largest singular value, estimated by power iteration. |
| `causal` | bool | `False` | — | Left-only causal padding; when stride=1 establishes bijective L_out == L_in. |
| `channels_last` | bool | `False` | — | Sequence layout [B, T, C] instead of [B, C, L]. |

## Description

Cross-correlation of the input with `out_channels` learned kernels of
shape `[in_channels / groups, kernel_size]`:

```text
out[b, co, l] = bias[co] + sum_{ci, k} weight[co, ci, k] * x[b, ci, l*stride - padding + k*dilation]
```

## Axis convention

The default input is `[B, C, L]`: channels at axis 1, positions at axis 2.
With `channels_last=True`, the input layout is `[B, T, C]`.
With `causal=True`, left-padding of `dilation * (kernel_size - 1)` is applied
so each position attends only to past and present positions.
When `causal=True` and `stride=1`, `L_out == L_in` forms a bijective relation.

## Examples

### Example 1

Padding 1 with kernel 3 preserves the length axis.

```python
conv1d(16, kernel_size=3, padding=1)
relu()
conv1d(4, kernel_size=3, padding=1)
```

Input `['B', 4, 32]` → output `['B', 4, 32]`.

```text
Network: [B, 4, 32] -> [B, 4, 32]  dtype=float32
index  name  operation  input shape  output shape
0      n0    conv1d     [B, 4, 32]   [B, 16, 32]
1      n1    relu       [B, 16, 32]  [B, 16, 32]
2      n2    conv1d     [B, 16, 32]  [B, 4, 32]
```

Parameters: 404

### Example 2

Kernel 4, stride 2, padding 1 halves the length.

```python
conv1d(8, kernel_size=4, stride=2, padding=1)
```

Input `['B', 3, 16]` → output `['B', 8, 8]`.

```text
Network: [B, 3, 16] -> [B, 8, 8]  dtype=float32
index  name  operation  input shape  output shape
0      n0    conv1d     [B, 3, 16]   [B, 8, 8]
```

Parameters: 104

### Example 3

Dilation 2 widens the receptive field to 5 positions and trims 4 from the length.

```python
conv1d(6, kernel_size=3, dilation=2)
group_norm(3)
relu()
```

Input `['B', 2, 16]` → output `['B', 6, 12]`.

```text
Network: [B, 2, 16] -> [B, 6, 12]  dtype=float32
index  name  operation   input shape  output shape
0      n0    conv1d      [B, 2, 16]   [B, 6, 12]
1      n1    group_norm  [B, 6, 12]   [B, 6, 12]
2      n2    relu        [B, 6, 12]   [B, 6, 12]
```

Parameters: 54

### Example 4

A spectrally normalized waveform discriminator trunk.

```python
conv1d(16, kernel_size=4, stride=2, padding=1, spectral_norm=True)
leaky_relu(0.2)
conv1d(32, kernel_size=4, stride=2, padding=1, spectral_norm=True)
```

Input `['B', 1, 64]` → output `['B', 32, 16]`.

```text
Network: [B, 1, 64] -> [B, 32, 16]  dtype=float32
index  name  operation   input shape  output shape
0      n0    conv1d      [B, 1, 64]   [B, 16, 32]
1      n1    leaky_relu  [B, 16, 32]  [B, 16, 32]
2      n2    conv1d      [B, 16, 32]  [B, 32, 16]
```

Parameters: 2,160

### Example 5

Causal convolution preserves sequence length bijectively when stride=1.

```python
conv1d(32, kernel_size=4, causal=True)
```

Input `['B', 16, 64]` → output `['B', 32, 64]`.

```text
Network: [B, 16, 64] -> [B, 32, 64]  dtype=float32
index  name  operation  input shape  output shape
0      n0    conv1d     [B, 16, 64]  [B, 32, 64]
```

Parameters: 2,080

### Example 6

Causal convolution directly on [B, T, C] sequence layout.

```python
conv1d(32, kernel_size=4, causal=True, channels_last=True)
```

Input `['B', 64, 16]` → output `['B', 64, 32]`.

```text
Network: [B, 64, 16] -> [B, 64, 32]  dtype=float32
index  name  operation  input shape  output shape
0      n0    conv1d     [B, 64, 16]  [B, 64, 32]
```

Parameters: 2,080
