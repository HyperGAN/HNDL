# `split`

Cut one axis into a first section and the remainder.

**Category:** shape · **Identity:** `split@1`

## Shape

```text
x -> first, rest
```

Relation: `first[dim] == size; rest[dim] == x[dim] - size; all other axes equal`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `first` | output | `first` | compute |
| `rest` | output | `rest` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `size` (positional) | int | inferred | >= 1; <= 2147483647 | Extent of the first section. Inferred when the consumers determine it. |
| `dim` | int | `1` | >= 1 | Axis to split; the batch axis 0 cannot be split. |

## Description

Returns ``(x[:size], x[size:])`` along ``dim`` as views sharing autograd
with the input. Exactly two non-empty sections are produced; this is not
a repeated chunking. After ``split`` there is no single current tensor, so
the next operation must name its input.

## Examples

### Example 1

Both sections feed explicit branches; split clears the current tensor.

```python
z1, z2 = split(64)
linear(z1, 32)
features = relu()
style = linear(z2, 32)
add(features, style)
```

Input `['B', 128]` → output `['B', 32]`.

```text
Network: [B, 128] -> [B, 32]  dtype=float32
index  name  operation  input shapes          output shapes
0      n0    split      x=[B, 128]            first=[B, 64], rest=[B, 64]
1      n1    linear     x=[B, 64]             out=[B, 32]
2      n2    relu       x=[B, 32]             out=[B, 32]
3      n3    linear     x=[B, 64]             out=[B, 32]
4      n4    add        a=[B, 32], b=[B, 32]  out=[B, 32]
```

Parameters: 4,160

### Example 2

The first size 96 is inferred from the remainder.

```python
a, b = split()
out = b
```

Input `['B', 128]` → output `['B', 32]`.

```text
Network: [B, 128] -> [B, 32]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    split      x=[B, 128]    first=[B, 96], rest=[B, 32]
```

Parameters: 0
