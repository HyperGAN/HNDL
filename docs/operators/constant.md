# `constant`

Emit a tensor of a fixed shape filled with one constant value.

**Category:** arithmetic · **Identity:** `constant@1`

## Shape

```text
x[B, ...]:any -> out
```

Relation: `out == [B, *shape]; x contributes only its batch extent and device`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]:any` | any |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `shape` (positional) | ints | `()` | >= 1; <= 2147483647 | Non-batch dimensions of the constant. Omit it to read them from the output contract. |
| `value` | float | `0.0` | — | Value every element is filled with. |

Positional values fill `shape`.

## Description

``out = full([B, *shape], value)``: a source of fixed values with no
parameters and no buffers.

```text
out[b, ...] = value
```

Give the non-batch dimensions positionally, ``constant(3, 8, 8)``, or as
``shape=(3, 8, 8)``; omit them entirely and the plan reads them from the
output contract, the way ``reshape()`` does. ``value`` defaults to ``0.0``,
so ``constant(16)`` is a zero vector — a null conditioning input — and
``constant(16, value=1.0)`` is a vector of ones.

The incoming tensor ``x`` supplies only the runtime batch extent and the
device; its own shape, rank, and values are ignored, so the input port
accepts any dtype including an integer graph input such as token ids. The
result always carries the plan's compute dtype and it is created fresh on
every call, detached from the autograd graph: it requires no gradient and
no gradient reaches ``x``. The operation is deterministic and identical in
train and eval mode.

Because the constant tells the solver nothing about ``x``, it does not
propagate shapes backward: the operation before it must have its shape
fixed from the input side or by its own arguments.

## Examples

### Example 1

A null conditioning vector of 16 zeros appended to the features.

```python
z = constant(16)
concat(x, z)
linear(8)
```

Input `['B', 4]` → output `['B', 8]`.

```text
Network: [B, 4] -> [B, 8]  dtype=float32
index  name  operation  input shapes           output shapes
0      n0    constant   x=[B, 4]               out=[B, 16]
1      n1    concat     x0=[B, 4], x1=[B, 16]  out=[B, 20]
2      n2    linear     x=[B, 20]              out=[B, 8]
```

Parameters: 168

### Example 2

A constant image plane: every pixel of every channel is centered by 0.5.

```python
c = constant(3, 8, 8, value=0.5)
sub(x, c)
conv(4, kernel_size=3, padding=1)
```

Input `['B', 3, 8, 8]` → output `['B', 4, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 4, 8, 8]  dtype=float32
index  name  operation  input shapes                    output shapes
0      n0    constant   x=[B, 3, 8, 8]                  out=[B, 3, 8, 8]
1      n1    sub        a=[B, 3, 8, 8], b=[B, 3, 8, 8]  out=[B, 3, 8, 8]
2      n2    conv       x=[B, 3, 8, 8]                  out=[B, 4, 8, 8]
```

Parameters: 112

### Example 3

The omitted shape is read from the [B, 8] contract the add imposes.

```python
h = linear(8)
z = constant(value=1.0)
add(h, z)
```

Input `['B', 4]` → output `['B', 8]`.

```text
Network: [B, 4] -> [B, 8]  dtype=float32
index  name  operation  input shapes        output shapes
0      n0    linear     x=[B, 4]            out=[B, 8]
1      n1    constant   x=[B, 8]            out=[B, 8]
2      n2    add        a=[B, 8], b=[B, 8]  out=[B, 8]
```

Parameters: 40
