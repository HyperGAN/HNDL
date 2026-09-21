# `reshape`

View the tensor with new non-batch dimensions, preserving the element count.

**Category:** shape · **Identity:** `reshape@1`

## Shape

```text
x -> out
```

Relation: `prod(x[1:]) == prod(out[1:]); out[1:len(shape)+1] == shape`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `shape` (positional) | ints | `()` | >= 1; <= 2147483647 | Leading non-batch dimensions. Remaining dimensions are inferred. |

Positional values fill `shape`.

## Description

Returns ``x.reshape(batch, *shape)``. The batch axis is never reshaped.
Give the leading dimensions positionally, ``reshape(512, 4, 4)``, or as
``shape=(512, 4, 4)``; exactly one omitted factor can be solved from the
element count. The plan records the full resolved shape.

## Examples

### Example 1

The projection width 128 follows from the reshape target.

```python
linear()
reshape(8, 4, 4)
```

Input `['B', 16]` → output `['B', 8, 4, 4]`.

```text
Network: [B, 16] -> [B, 8, 4, 4]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 16]     out=[B, 128]
1      n1    reshape    x=[B, 128]    out=[B, 8, 4, 4]
```

Parameters: 2,176

### Example 2

Height and width are inferred from the element count and the output contract.

```python
reshape(512)
conv(3, kernel_size=1)
```

Input `['B', 8192]` → output `['B', 3, 4, 4]`.

```text
Network: [B, 8192] -> [B, 3, 4, 4]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    reshape    x=[B, 8192]       out=[B, 512, 4, 4]
1      n1    conv       x=[B, 512, 4, 4]  out=[B, 3, 4, 4]
```

Parameters: 1,539

### Example 3

A bare reshape flattens when the consumer fixes rank 2.

```python
reshape()
linear()
```

Input `['B', 2, 4, 4]` → output `['B', 10]`.

```text
Network: [B, 2, 4, 4] -> [B, 10]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    reshape    x=[B, 2, 4, 4]  out=[B, 32]
1      n1    linear     x=[B, 32]       out=[B, 10]
```

Parameters: 330
