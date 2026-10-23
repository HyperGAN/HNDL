# `snake`

Periodic activation: x + (1 / (alpha + eps)) * sin^2(alpha * x).

**Category:** activation · **Identity:** `snake@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `channels` (positional) | int | `0` | >= 0 | Number of channels C; 0 infers from input. |
| `dim` | int | `0` | >= 0 | Alias for channels. |
| `alpha` | float | `1.0` | — | Initial frequency scale parameter alpha. |
| `eps` | float | `1e-09` | > 0.0 | Epsilon to prevent division by zero. |

## Description

Periodic activation function for neural audio and inductive bias.

## Examples

### Example 1

Periodic activation with learnable frequency parameter per channel.

```python
snake(16)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
index  name  operation  input shape  output shape
0      n0    snake      [B, 16, 64]  [B, 16, 64]
```

Parameters: 16
