# `sinusoidal_embed`

Deterministic log-spaced sinusoidal timestep embeddings.

**Category:** sequence · **Identity:** `sinusoidal_embed@1`

## Shape

```text
x[B, D_in] -> out[B, D_out]
```

Relation: `D_out = features (must be even and a multiple of 2 * D_in)`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, D_in]` | compute |
| `out` | output | `out[B, D_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `features` (positional) | int | inferred | >= 1; binds `D_out` | Output embedding dimension. |
| `dim` | int | `0` | >= 0 | Alias for features. |
| `max_period` | float | `10000.0` | > 1.0 | Maximum frequency period for log-spaced frequencies. |

## Description

Deterministic log-spaced sinusoidal timestep and coordinate embedding.

Computes frequency bands geometrically spaced from 1 down to 1 / max_period,
mapping inputs into interleaved or concatenated sine and cosine components:
``angles = x * freqs``
``out = [sin(angles), cos(angles)]``.

## Examples

### Example 1

Embeds scalar timesteps into 64 sinusoidal features.

```python
sinusoidal_embed(64)
```

Input `['B', 1]` → output `['B', 64]`.

```text
Network: [B, 1] -> [B, 64]  dtype=float32
index  name  operation         input shape  output shape
0      n0    sinusoidal_embed  [B, 1]       [B, 64]
```

Parameters: 0
