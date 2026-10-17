# `cast`

Explicit dtype conversion across graph edges.

**Category:** core · **Identity:** `cast@1`

## Shape

```text
x[B, ...]:any -> out[B, ...]:any
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]:any` | any |
| `out` | output | `out[B, ...]:any` | any |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dtype` (positional) | str | `"compute"` | one of `compute`, `float32`, `float16`, `bfloat16`, `float64`, `int64`, `int32`, `bool` | Target tensor dtype; 'compute' targets the plan compute dtype. |

## Description

Cast a tensor to a target data type across graph edges.

Converts an input tensor along the edge to the specified target data type.
When ``dtype`` is set to ``"compute"``, the tensor is cast to the active
plan compute dtype (e.g. ``float32``, ``float16``, or ``bfloat16``).

## Examples

### Example 1

Cast features to the plan compute dtype.

```python
cast('compute')
```

Input `['B', 16]` → output `['B', 16]`.

```text
Network: [B, 16] -> [B, 16]  dtype=float32
index  name  operation  input shape  output shape
0      n0    cast       [B, 16]      [B, 16]
```

Parameters: 0
