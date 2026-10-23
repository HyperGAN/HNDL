# `rope`

Standalone rotary position embedding over a [B, T, D] or [B, H, T, D_h] sequence.

**Category:** sequence · **Identity:** `rope@1`

## Shape

```text
x[B, T, D] -> out[B, T, D]
```

Relation: `D % heads == 0 and (D // heads) % 2 == 0; out matches x`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `heads` (positional) | int | `1` | >= 1 | Number of heads to split D into before rotating. |
| `theta` | float | `10000.0` | > 1.0 | Base frequency for rotary embeddings. |
| `rope_scaling` | str | `"none"` | one of `none`, `linear`, `ntk`, `yarn`, `llama3` | Frequency scaling algorithm for RoPE context extension. |
| `rope_scale` | float | `1.0` | > 0.0 | Frequency scale factor for context extension. |
| `spatial_shape` | pair | `(0, 0)` | >= 0 | Token grid (H, W) for 2D axial RoPE, row-major with H*W == T. |

## Description

Standalone rotary position embedding (RoPE) for [B, T, D] or [B, H, T, D_h] sequences.

Applies rotary position embeddings independently to each head of dimension
``head_dim = D // heads`` along the sequence length axis:
``out = x * cos + rotate_half(x) * sin``.

## Examples

### Example 1

Rotates 4 heads of dimension 16 along the length axis.

```python
linear(64)
rope(4)
linear()
```

Input `['B', 16, 32]` → output `['B', 16, 10]`.

```text
Network: [B, 16, 32] -> [B, 16, 10]  dtype=float32
index  name  operation  input shape  output shape
0      n0    linear     [B, 16, 32]  [B, 16, 64]
1      n1    rope       [B, 16, 64]  [B, 16, 64]
2      n2    linear     [B, 16, 64]  [B, 16, 10]
```

Parameters: 2,762
