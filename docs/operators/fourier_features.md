# `fourier_features`

Encode coordinates with a fixed random Fourier frequency table.

**Category:** sequence · **Identity:** `fourier_features@1`

## Shape

```text
x[B, ..., D] -> out[B, ..., 2*F]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ..., D]` | compute |
| `out` | output | `out[B, ..., 2*F]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `num_frequencies` (positional) | int | required | >= 1; <= 1073741823; binds `F` | Number of frequencies; output width is twice this value. |
| `scale` | float | `8.0` | > 0 | Standard deviation of the fixed normal frequency table. |

## Description

With fixed ``frequencies ~ Normal(0, scale)``, compute
``p = x @ frequencies.T`` and ``concat(sin(p), cos(p), dim=-1)``.
No extra 2*pi factor is applied. Frequencies have shape [F,D] and are
a persistent buffer, not trainable parameters. Initialization uses the
construction RNG; forward draws nothing. To share one table between
two sets of coordinates, concatenate them before this operator and
split their features afterwards. Gradients flow to the coordinates.

## Examples

### Example 1

Encode 2D coordinates with eight sine and eight cosine features.

```python
fourier_features(8, scale=8.0)
```

Input `['B', 16, 2]` → output `['B', 16, 16]`.

```text
Network: [B, 16, 2] -> [B, 16, 16]  dtype=float32
index  name  operation         input shapes  output shapes
0      n0    fourier_features  x=[B, 16, 2]  out=[B, 16, 16]
```

Parameters: 0
