# `conv`

Two-dimensional convolution over [B, C, H, W] images.

**Category:** convolution · **Identity:** `conv2d@1`

## Shape

```text
x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]
```

Relation: `H_out = floor((H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1), same for W`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C_in, H_in, W_in]` | compute |
| `out` | output | `out[B, C_out, H_out, W_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `out_channels` (positional) | int | inferred | >= 1; <= 2147483647 | Output channels. Omit to infer from the consumer. |
| `in_channels` | int | inferred | >= 1; <= 2147483647 | Input channels. Normally inferred from the incoming tensor. |
| `kernel_size` | pair | required | >= 1 | Kernel height and width; an int applies to both. |
| `stride` | pair | `(1, 1)` | >= 1 | Step between kernel applications. |
| `padding` | pair | `(0, 0)` | >= 0 | Zero padding added to each spatial side. |
| `dilation` | pair | `(1, 1)` | >= 1 | Spacing between kernel taps. |
| `groups` | int | `1` | >= 1 | Channel groups; must divide input and output channels. |
| `bias` | bool | `True` | — | Add a learned per-channel bias. |

## Description

Cross-correlation of the input with ``out_channels`` learned kernels of
shape ``[in_channels / groups, kH, kW]``. Inverse inference from a known
output extent may leave an interval of valid input sizes; that ambiguity
is reported rather than resolved arbitrarily.

## Examples

### Example 1

Padding 1 with kernel 3 preserves height and width.

```python
conv(16, kernel_size=3, padding=1)
relu()
conv(3, kernel_size=3, padding=1)
```

Input `['B', 3, 32, 32]` → output `['B', 3, 32, 32]`.

```text
Network: [B, 3, 32, 32] -> [B, 3, 32, 32]  dtype=float32
index  name  operation  input shapes       output shapes
0      n0    conv       x=[B, 3, 32, 32]   out=[B, 16, 32, 32]
1      n1    relu       x=[B, 16, 32, 32]  out=[B, 16, 32, 32]
2      n2    conv       x=[B, 16, 32, 32]  out=[B, 3, 32, 32]
```

Parameters: 883

### Example 2

Kernel 4, stride 2, padding 1 halves each spatial axis.

```python
conv(8, kernel_size=4, stride=2, padding=1)
```

Input `['B', 3, 16, 16]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 16, 16] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    conv       x=[B, 3, 16, 16]  out=[B, 8, 8, 8]
```

Parameters: 392
