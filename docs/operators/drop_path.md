# `drop_path`

Stochastic depth (DropPath) residual regularization.

**Category:** regularization · **Identity:** `stochastic_depth@1`

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
| `drop_prob` (positional) | float | `0.0` | >= 0.0; <= 1.0 | Probability of dropping residual path. |
| `scale_by_keep` | bool | `True` | — | Scale retained samples by 1 / (1 - drop_prob). |

## Description

Stochastic depth layer (DropPath) per sample across batches.

## Examples

### Example 1

Stochastic depth dropping residual connections (drop_prob=0.0 in examples for determinism).

```python
drop_path(0.0)
```

Input `['B', 8, 32]` → output `['B', 8, 32]`.

```text
Network: [B, 8, 32] -> [B, 8, 32]  dtype=float32
index  name  operation  input shape  output shape
0      n0    drop_path  [B, 8, 32]   [B, 8, 32]
```

Parameters: 0
