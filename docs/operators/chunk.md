# `chunk`

Cut one axis into a fixed number of equal sections.

**Category:** shape · **Identity:** `chunk@1`

## Shape

```text
x -> out*
```

Relation: `out_i[dim] == x[dim] / chunks for every i; all other axes equal`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out*` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `chunks` (positional) | int | required | >= 1; <= 32 | Number of equal sections, which is also the number of outputs. |
| `dim` | int | `1` | >= 0 | Axis to cut; 0 divides the batch, which must be a multiple of chunks. |

## Description

Returns ``chunks`` equal sections of ``dim`` as views sharing autograd
with the input, in order, from ``torch.split`` with one exact size.

``chunk`` is the equal-split counterpart of ``split``, which cuts one
tensor into a first section and the remainder. The axis must divide
exactly: an extent of 9 into 2 sections fails with ``E_CONSTRAINT`` at
resolution, not at runtime, and no section may be empty.

``dim=0`` divides the batch. Batch is symbolic, so the division is done on
batch entries: a ``2*B`` input --- what ``concat(..., axis=0)`` of two
``B`` tensors produces --- chunks into two ``B`` outputs, and ``3*B`` into
three. A plain ``B`` input cannot be chunked, because nothing says the
runtime batch is even; that fails at resolution with ``E_CONSTRAINT``. The
module still checks divisibility in ``forward`` (``E_RUNTIME``) for a
submodule called directly, outside the graph that checked its contract::

    combined = concat(candidate, context, axis=0)   # [2*B, C, H, W]
    features = pretrained(combined, ...)            # one shared pass
    a, b = chunk(features, 2, dim=0)                # [B, ...] and [B, ...]

Like every multi-output call, ``chunk`` returns a tuple and clears the
current tensor, so the next operation must name its input. ``chunks=1`` is
allowed and returns a one-tuple containing the whole tensor, which also
clears current. Both first and second derivatives flow through every
section: the sections are views, not copies.

## Examples

### Example 1

Two equal halves of the feature axis, concatenated back in the other order.

```python
a, b = chunk(2)
concat(b, a)
```

Input `['B', 8]` → output `['B', 8]`.

```text
Network: [B, 8] -> [B, 8]  dtype=float32
index  name  operation  input shapes          output shapes
0      n0    chunk      x=[B, 8]              out0=[B, 4], out1=[B, 4]
1      n1    concat     x0=[B, 4], x1=[B, 4]  out=[B, 8]
```

Parameters: 0

### Example 2

One linear runs over both copies at batch 2*B; chunk along dim 0 returns the pair at batch B.

```python
pair = concat(x, x, axis=0)
shared = linear(pair, 6)
p, q = chunk(shared, 2, dim=0)
concat(p, q)
```

Input `['B', 4]` → output `['B', 12]`.

```text
Network: [B, 4] -> [B, 12]  dtype=float32
index  name  operation  input shapes          output shapes
0      n0    concat     x0=[B, 4], x1=[B, 4]  out=[2*B, 4]
1      n1    linear     x=[2*B, 4]            out=[2*B, 6]
2      n2    chunk      x=[2*B, 6]            out0=[B, 6], out1=[B, 6]
3      n3    concat     x0=[B, 6], x1=[B, 6]  out=[B, 12]
```

Parameters: 30

### Example 3

Three equal sections of a rank-3 tensor, reordered.

```python
a, b, c = chunk(3, dim=2)
concat(a, c, b, axis=2)
```

Input `['B', 4, 9]` → output `['B', 4, 9]`.

```text
Network: [B, 4, 9] -> [B, 4, 9]  dtype=float32
index  name  operation  input shapes                              output shapes
0      n0    chunk      x=[B, 4, 9]                               out0=[B, 4, 3], out1=[B, 4, 3], out2=[B, 4, 3]
1      n1    concat     x0=[B, 4, 3], x1=[B, 4, 3], x2=[B, 4, 3]  out=[B, 4, 9]
```

Parameters: 0
