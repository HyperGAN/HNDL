# `leaky_relu`

ReLU with a small slope for negative inputs.

**Category:** activation · **Identity:** `leaky_relu@1`

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
| `negative_slope` (positional) | float | `0.01` | — | Multiplier applied to negative inputs. |

## Description

``out = x if x >= 0 else negative_slope * x``, computed out of place.

## Examples

### Example 1

```python
conv(8, kernel_size=3, padding=1)
leaky_relu(0.2)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation   input shapes    output shapes
0      n0    conv        x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    leaky_relu  x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224
