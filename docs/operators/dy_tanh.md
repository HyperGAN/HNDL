# `dy_tanh`

Dynamic Tanh activation with learned scalar alpha and affine scaling.

**Category:** activation · **Identity:** `dynamic_tanh@1`

## Shape

```text
x[B, ..., D] -> out[B, ..., D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ..., D]` | compute |
| `out` | output | `out[B, ..., D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dim` (positional) | int | inferred | >= 1; binds `D` | Feature dimension D. |
| `init_alpha` | float | `1.0` | — | Initial value for learned scalar alpha. |
| `affine` | bool | `True` | — | Learn affine per-channel gamma and beta. |

## Description

Dynamic Tanh activation layer: y = gamma * tanh(alpha * x) + beta.

## Examples

### Example 1

Dynamic Tanh normalization with learned alpha and affine gamma/beta.

```python
dy_tanh(dim=32)
```

Input `['B', 8, 32]` → output `['B', 8, 32]`.

```text
Network: [B, 8, 32] -> [B, 8, 32]  dtype=float32
index  name  operation  input shape  output shape
0      n0    dy_tanh    [B, 8, 32]   [B, 8, 32]
```

Parameters: 65
