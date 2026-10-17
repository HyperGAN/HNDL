# `grn`

Global Response Normalization (GRN) for ConvNeXt-V2.

**Category:** normalization · **Identity:** `grn@1`

## Shape

```text
x[B, C, H, W] -> out[B, C, H, W]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H, W]` | compute |
| `out` | output | `out[B, C, H, W]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `eps` | float | `1e-06` | > 0 | Small positive constant to avoid division by zero. |

## Description

Global Response Normalization (GRN) for ConvNeXt-V2.

Computes spatial L2 norm per channel, normalizes across the channel axis,
and recalibrates features via zero-initialized learned scale and bias:
``Gx = sqrt(sum(x^2, dim=(2, 3)) + eps)``
``Nx = Gx / (mean(Gx, dim=1) + eps)``
``out = gamma * (x * Nx) + beta + x``.

## Examples

### Example 1

Applies global response normalization across feature channels.

```python
conv(32, kernel_size=3, padding=1)
grn()
conv(10, kernel_size=1)
```

Input `['B', 16, 8, 8]` → output `['B', 10, 8, 8]`.

```text
Network: [B, 16, 8, 8] -> [B, 10, 8, 8]  dtype=float32
index  name  operation  input shape    output shape
0      n0    conv       [B, 16, 8, 8]  [B, 32, 8, 8]
1      n1    grn        [B, 32, 8, 8]  [B, 32, 8, 8]
2      n2    conv       [B, 32, 8, 8]  [B, 10, 8, 8]
```

Parameters: 5,034
