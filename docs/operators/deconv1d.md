# `deconv1d`

Transposed one-dimensional convolution for waveform synthesis and upsampling.

**Category:** convolution · **Identity:** `conv_transpose1d@1`

## Shape

```text
x[B, C_in, L_in] -> out[B, C_out, L_out]
```

Relation: `L_out = (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C_in, L_in]` | compute |
| `out` | output | `out[B, C_out, L_out]` | compute |

**Policies:** `policy="up2"` (spatial1d.up2_transpose@1: kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0, groups=1)

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `out_channels` (positional) | int | inferred | >= 1; <= 2147483647 | Output channels. Omit to infer from the consumer. |
| `in_channels` | int | inferred | >= 1; <= 2147483647 | Input channels. Normally inferred from the incoming tensor. |
| `kernel_size` | int | required | >= 1 | Kernel length in positions along the length axis. |
| `stride` | int | `1` | >= 1 | Upsampling step along the length axis. |
| `padding` | int | `0` | >= 0 | Implicit zero padding removed from each side of the output. |
| `dilation` | int | `1` | >= 1 | Spacing between kernel taps. |
| `output_padding` | int | `0` | >= 0 | Extra size added to one side of the output length axis. |
| `groups` | int | `1` | >= 1 | Channel groups; must divide input and output channels. |
| `bias` | bool | `True` | — | Add a learned per-channel bias. |
| `spectral_norm` | bool | `False` | — | Divide the weight by its largest singular value, estimated by power iteration. |

## Description

The gradient of ``conv1d`` with respect to its input, used as a learned
upsampling layer along the length axis of ``[B, C, L]`` signals.

Select ``policy="up2"`` to guarantee that length doubles exactly.
Parameters are ``weight`` of shape ``[in_channels, out_channels / groups, kernel_size]``
and, when ``bias=True``, ``bias`` of shape ``[out_channels]``.

## Examples

### Example 1

The "up2" policy selects kernel 4, stride 2, padding 1: exact length doubling.

```python
deconv1d(64, policy="up2")
relu()
deconv1d(1, policy="up2")
```

Input `['B', 128, 16]` → output `['B', 1, 64]`.

```text
Network: [B, 128, 16] -> [B, 1, 64]  dtype=float32
index  name  operation  input shape   output shape
0      n0    deconv1d   [B, 128, 16]  [B, 64, 32]
1      n1    relu       [B, 64, 32]   [B, 64, 32]
2      n2    deconv1d   [B, 64, 32]   [B, 1, 64]
```

Parameters: 33,089

### Example 2

Explicit kernel parameters double the length axis.

```python
deconv1d(16, kernel_size=4, stride=2, padding=1)
```

Input `['B', 8, 32]` → output `['B', 16, 64]`.

```text
Network: [B, 8, 32] -> [B, 16, 64]  dtype=float32
index  name  operation  input shape  output shape
0      n0    deconv1d   [B, 8, 32]   [B, 16, 64]
```

Parameters: 528

### Example 3

Spectrally normalized 1D upsampling layer for audio generators.

```python
deconv1d(32, policy="up2", spectral_norm=True)
```

Input `['B', 16, 8]` → output `['B', 32, 16]`.

```text
Network: [B, 16, 8] -> [B, 32, 16]  dtype=float32
index  name  operation  input shape  output shape
0      n0    deconv1d   [B, 16, 8]   [B, 32, 16]
```

Parameters: 2,080
