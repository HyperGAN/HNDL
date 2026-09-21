# `flatten`

Collapse every non-batch axis into one feature axis.

**Category:** shape · **Identity:** `flatten@1`

## Shape

```text
x[B, ...] -> out[B, F]
```

Relation: `F == prod(x[1:])`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, F]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Equivalent to ``x.reshape(batch, -1)``. The input rank can be inferred
backward only through the element count when the other axes are known.

## Examples

### Example 1

```python
flatten()
linear()
```

Input `['B', 3, 8, 8]` → output `['B', 2]`.

```text
Network: [B, 3, 8, 8] -> [B, 2]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    flatten    x=[B, 3, 8, 8]  out=[B, 192]
1      n1    linear     x=[B, 192]      out=[B, 2]
```

Parameters: 386
