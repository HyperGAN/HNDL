# `joint_attention`

Dual-stream MMDiT joint self-attention across image and context streams.

**Category:** sequence · **Identity:** `mmdit_attention@1`

## Shape

```text
x[B, T_x, D_x], context[B, T_c, D_c] -> out[B, T_x, D_x], ctx_out[B, T_c, D_c]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T_x, D_x]` | compute |
| `context` | input | `context[B, T_c, D_c]` | compute |
| `out` | output | `out[B, T_x, D_x]` | compute |
| `ctx_out` | output | `ctx_out[B, T_c, D_c]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `heads` (positional) | int | required | >= 1 | Number of joint attention heads. |
| `head_dim` | int | `0` | >= 0 | Per-head dimension; 0 defaults to D // heads. |
| `qk_norm` | str | `"none"` | one of `none`, `rms_norm`, `l2_norm` | QK normalization. |
| `bias` | bool | `True` | — | Whether linear projections have bias. |
| `dropout` | float | `0.0` | >= 0.0; < 1.0 | Dropout probability. |
| `eps` | float | `1e-06` | > 0.0 | Epsilon for RMSNorm. |

## Description

Dual-stream Joint Attention (MMDiT / Flux) operator.

## Examples

### Example 1

MMDiT dual-stream joint attention block.

```python
c = linear(x, 32)
x1, c1 = joint_attention(x, c, heads=4, head_dim=16)
add(x1, c1)
```

Input `['B', 8, 32]` → output `['B', 8, 32]`.

```text
Network: [B, 8, 32] -> [B, 8, 32]  dtype=float32
name  operation        input shapes                                            output shapes
n0    linear           x=input:x:[B, 8, 32]                                    out=[B, 8, 32]
n1    joint_attention  x=input:x:[B, 8, 32], context=node:n0/out:[B, 8, 32]    out=[B, 8, 32], ctx_out=[B, 8, 32]
n2    add              a=node:n1/out:[B, 8, 32], b=node:n1/ctx_out:[B, 8, 32]  out=[B, 8, 32]
```

Parameters: 17,888
