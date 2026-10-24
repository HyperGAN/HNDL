# `mamba2`

State Space Duality (SSD) block from Mamba-2.

**Category:** sequence · **Identity:** `ssd@1`

## Shape

```text
x[B, T, D] -> out[B, T, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `d_model` (positional) | int | inferred | >= 1; binds `D` | Model feature dimension D. |
| `d_state` | int | `64` | >= 1 | State space dimension N per head. |
| `d_conv` | int | `4` | >= 1 | 1D depthwise convolution width. |
| `expand` | int | `2` | >= 1 | Inner dimension expansion factor. |
| `heads` | int | `8` | >= 1 | Number of heads. |
| `head_dim` | int | `64` | >= 1 | Head dimension. |
| `chunk_size` | int | `64` | >= 1 | Chunk size for semi-separable recurrence. |
| `bias` | bool | `False` | — | Whether linear projections have bias. |
| `conv_bias` | bool | `True` | — | Whether depthwise conv1d has bias. |

## Description

Mamba-2 State Space Duality (SSD) layer.

## Examples

### Example 1

Mamba-2 SSD sequence transformation block.

```python
mamba2(d_model=32, d_state=16, heads=4, head_dim=16)
```

Input `['B', 8, 32]` → output `['B', 8, 32]`.

```text
Network: [B, 8, 32] -> [B, 8, 32]  dtype=float32
index  name  operation  input shape  output shape
0      n0    mamba2     [B, 8, 32]   [B, 8, 32]
```

Parameters: 11,404
