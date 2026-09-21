# `concat`

Join two or more tensors along one axis.

**Category:** shape · **Identity:** `concat@1`

## Shape

```text
x* -> out
```

Relation: `out[axis] == sum(x_i[axis]); all other axes equal`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x*` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `input_count` | int | required | >= 2 | Number of tensors joined; set from the call. |
| `axis` | int | `1` | >= 1 | Axis to concatenate; the batch axis 0 is excluded. |

## Description

``torch.cat(inputs, dim=axis)``. Every input is explicit; one missing
extent along ``axis`` can be inferred from the output contract.

## Examples

### Example 1

Reorder sections by concatenating them back.

```python
a, b = split(2)
concat(b, a)
```

Input `['B', 5]` → output `['B', 5]`.

```text
Network: [B, 5] -> [B, 5]  dtype=float32
index  name  operation  input shapes          output shapes
0      n0    split      x=[B, 5]              first=[B, 2], rest=[B, 3]
1      n1    concat     x0=[B, 3], x1=[B, 2]  out=[B, 5]
```

Parameters: 0

### Example 2

The second width 6 is inferred.

```python
a = linear(4)
b = linear(x)
concat(a, b)
```

Input `['B', 8]` → output `['B', 10]`.

```text
Network: [B, 8] -> [B, 10]  dtype=float32
index  name  operation  input shapes          output shapes
0      n0    linear     x=[B, 8]              out=[B, 4]
1      n1    linear     x=[B, 8]              out=[B, 6]
2      n2    concat     x0=[B, 4], x1=[B, 6]  out=[B, 10]
```

Parameters: 90
