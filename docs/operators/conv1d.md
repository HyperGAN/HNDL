# `conv1d`

One-dimensional convolution over [B, C, L] signals.

**Category:** convolution · **Identity:** `conv1d@1`

## Shape

```text
x[B, C_in, L_in] -> out[B, C_out, L_out]
```

Relation: `L_out = floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C_in, L_in]` | compute |
| `out` | output | `out[B, C_out, L_out]` | compute |

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

## Description

Cross-correlation of the input with `out_channels` learned kernels of
shape `[in_channels / groups, kernel_size]`:

```text
out[b, co, l] = bias[co] + sum_{ci, k} weight[co, ci, k] * x[b, ci, l*stride - padding + k*dilation]
```

## Axis convention

The input is `[B, C, L]`: channels at axis 1, positions at axis 2. The
kernel slides along the length axis only. A `[B, T, D]` sequence carries
its features on the last axis instead, so transpose it to `[B, D, T]`
before this operator and back afterwards; `conv1d` never reinterprets the
axes for you.

With `groups > 1` the channels split into that many independent groups,
each convolved with its own kernels; `groups == in_channels ==
out_channels` is a depthwise convolution. Both channel counts must be
divisible by `groups`.

The output length is
`floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)`.
Because that formula is not injective, inferring the input length from a
known output length generally leaves an interval of valid values; the
resolver reports the ambiguity rather than choosing one.

Parameters are `weight` of shape `[out_channels, in_channels / groups,
kernel_size]` and, when enabled, `bias` of shape `[out_channels]`.
Behavior is identical in train and eval mode, and the computation stays in
the input dtype.

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
index  name  operation  input shapes   output shapes
0      n0    conv1d     x=[B, 4, 32]   out=[B, 16, 32]
1      n1    relu       x=[B, 16, 32]  out=[B, 16, 32]
2      n2    conv1d     x=[B, 16, 32]  out=[B, 4, 32]
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
index  name  operation  input shapes  output shapes
0      n0    conv1d     x=[B, 3, 16]  out=[B, 8, 8]
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
index  name  operation   input shapes  output shapes
0      n0    conv1d      x=[B, 2, 16]  out=[B, 6, 12]
1      n1    group_norm  x=[B, 6, 12]  out=[B, 6, 12]
2      n2    relu        x=[B, 6, 12]  out=[B, 6, 12]
```

Parameters: 54
