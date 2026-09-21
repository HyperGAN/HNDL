# `deconv`

Transposed two-dimensional convolution, typically for upsampling.

**Category:** convolution · **Identity:** `conv_transpose2d@1`

## Shape

```text
x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]
```

Relation: `H_out = (H_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1, same for W`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C_in, H_in, W_in]` | compute |
| `out` | output | `out[B, C_out, H_out, W_out]` | compute |

**Policies:** `policy="up2"` (spatial.up2_transpose@1: kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0, groups=1)

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `out_channels` (positional) | int | inferred | >= 1; <= 2147483647 | Output channels. Omit to infer from the consumer. |
| `in_channels` | int | inferred | >= 1; <= 2147483647 | Input channels. Normally inferred from the incoming tensor. |
| `kernel_size` | pair | required | >= 1 | Kernel height and width; an int applies to both. |
| `stride` | pair | `(1, 1)` | >= 1 | Upsampling step. |
| `padding` | pair | `(0, 0)` | >= 0 | Implicit zero padding removed from each side of the output. |
| `dilation` | pair | `(1, 1)` | >= 1 | Spacing between kernel taps. |
| `output_padding` | pair | `(0, 0)` | >= 0 | Extra size added to one side of each output axis. |
| `groups` | int | `1` | >= 1 | Channel groups; must divide input and output channels. |
| `bias` | bool | `True` | — | Add a learned per-channel bias. |

## Description

The gradient of ``conv`` with respect to its input, used as a learned
upsampling layer. Select ``policy="up2"`` to guarantee that height and
width double; explicit arguments that contradict the policy fail.

## Examples

### Example 1

The "up2" policy selects kernel 4, stride 2, padding 1: exact doubling.

```python
deconv(64, policy="up2")
relu()
deconv(3, policy="up2")
```

Input `['B', 128, 4, 4]` → output `['B', 3, 16, 16]`.

```text
Network: [B, 128, 4, 4] -> [B, 3, 16, 16]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    deconv     x=[B, 128, 4, 4]  out=[B, 64, 8, 8]
1      n1    relu       x=[B, 64, 8, 8]   out=[B, 64, 8, 8]
2      n2    deconv     x=[B, 64, 8, 8]   out=[B, 3, 16, 16]
```

Parameters: 134,211

### Example 2

The seed 4×4 and projection width 512 are inferred backward.

```python
linear()
reshape(32)
deconv(3, kernel_size=4, stride=2, padding=1)
```

Input `['B', 16]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 16] -> [B, 3, 8, 8]  dtype=float32
index  name  operation  input shapes     output shapes
0      n0    linear     x=[B, 16]        out=[B, 512]
1      n1    reshape    x=[B, 512]       out=[B, 32, 4, 4]
2      n2    deconv     x=[B, 32, 4, 4]  out=[B, 3, 8, 8]
```

Parameters: 10,243
