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
| `axis` | int | `1` | >= 0 | Axis to concatenate; axis 0 joins along batch and adds the inputs' batches. |

## Description

``torch.cat(inputs, dim=axis)``. Every input is explicit; one missing
extent along ``axis`` can be inferred from the output contract.

``axis=0`` joins along batch: the inputs must agree on every other axis and
the result carries their batches added together, written ``2*B`` for two
equal batches. That is how one shared --- often frozen --- network runs over
two branches in a single forward pass; ``chunk(..., dim=0)`` takes the
result apart again.

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

### Example 3

One shared linear runs over both branches at batch 2*B; chunk takes the pair apart.

```python
a = linear(4)
b = linear(x, 4)
pair = concat(a, b, axis=0)
shared = linear(pair, 6)
p, q = chunk(shared, 2, dim=0)
concat(p, q)
```

Input `['B', 8]` → output `['B', 12]`.

```text
Network: [B, 8] -> [B, 12]  dtype=float32
index  name  operation  input shapes          output shapes
0      n0    linear     x=[B, 8]              out=[B, 4]
1      n1    linear     x=[B, 8]              out=[B, 4]
2      n2    concat     x0=[B, 4], x1=[B, 4]  out=[2*B, 4]
3      n3    linear     x=[2*B, 4]            out=[2*B, 6]
4      n4    chunk      x=[2*B, 6]            out0=[B, 6], out1=[B, 6]
5      n5    concat     x0=[B, 6], x1=[B, 6]  out=[B, 12]
```

Parameters: 102
