# `linear`

Fully connected layer: a learned affine map on the last axis.

**Category:** core · **Identity:** `linear@1`

## Shape

```text
x[B, ..., D_in] -> out[B, ..., D_out]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ..., D_in]` | compute |
| `out` | output | `out[B, ..., D_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `out_features` (positional) | int | inferred | >= 1; <= 2147483647; binds `D_out` | Output width. Omit it to infer the width from what follows. |
| `in_features` | int | inferred | >= 1; <= 2147483647; binds `D_in` | Input width. Normally inferred from the incoming tensor. |
| `bias` | bool | `True` | — | Add a learned bias vector. |

## Description

Computes ``out = x @ weight.T + bias`` with ``weight`` of shape
``[out_features, in_features]``, applied to the last axis of ``[B, D]``
or ``[B, T, D]`` inputs. No activation is applied; add one explicitly.
Parameters are ``weight`` and, when enabled, ``bias``.

## Examples

### Example 1

The final width is inferred from the output contract.

```python
linear(64)
relu()
linear()
```

Input `['B', 128]` → output `['B', 10]`.

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 128]    out=[B, 64]
1      n1    relu       x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 8,906

### Example 2

```python
linear(32, bias=False)
```

Input `['B', 16]` → output `['B', 32]`.

```text
Network: [B, 16] -> [B, 32]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 16]     out=[B, 32]
```

Parameters: 512

### Example 3

On a [B, T, D] sequence the map applies to every position.

```python
linear(64)
relu()
linear()
```

Input `['B', 16, 32]` → output `['B', 16, 8]`.

```text
Network: [B, 16, 32] -> [B, 16, 8]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    linear     x=[B, 16, 32]  out=[B, 16, 64]
1      n1    relu       x=[B, 16, 64]  out=[B, 16, 64]
2      n2    linear     x=[B, 16, 64]  out=[B, 16, 8]
```

Parameters: 2,632
