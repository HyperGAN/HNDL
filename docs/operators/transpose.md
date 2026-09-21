# `transpose`

Swap two non-batch axes of the tensor.

**Category:** shape · **Identity:** `transpose@1`

## Shape

```text
x -> out
```

Relation: `out[dim0] == x[dim1]; out[dim1] == x[dim0]; every other axis is unchanged`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dim0` (positional) | int | required | >= 1; <= 3 | First axis of the swap; 1 is the first non-batch axis and 3 is the last axis of a rank-4 tensor. The batch axis 0 cannot be swapped. |
| `dim1` (positional) | int | required | >= 1; <= 3 | Second axis of the swap. Equal to dim0 means the identity. |

## Description

Returns ``x.transpose(dim0, dim1)``: the extents at ``dim0`` and
``dim1`` trade places and every other axis keeps its extent. The batch
axis 0 is never part of a swap, so both arguments are at least 1 and at
most 3 (the last axis of the deepest supported rank). Passing the same
axis twice is the identity.

Axis conventions. A rank-3 tensor is ``[B, T, D]`` for the sequence
operations — ``linear`` and the activations act on the last axis, ``D``,
across ``T`` positions — while 1-D convolution and pooling read rank 3 as
``[B, C, L]``, channels before length. ``transpose(1, 2)`` is the bridge
between the two readings, and a second ``transpose(1, 2)`` afterwards
returns to the sequence layout. At rank 4 the layout is ``[B, C, H, W]``,
so ``transpose(2, 3)`` swaps height and width and ``transpose(1, 3)``
exchanges channels with width.

The result is a view: it shares storage and autograd history with the
input and is generally not contiguous. Downstream operations handle
non-contiguous inputs; ``reshape`` and ``flatten`` copy when they must.
There are no parameters, no buffers, and no difference between train and
eval mode. The computation is a stride permutation, so it is exact in
every compute dtype.

## Examples

### Example 1

Bridges a [B, T, D] sequence to the [B, C, L] layout 1-D convolutions expect.

```python
transpose(1, 2)
```

Input `['B', 16, 32]` → output `['B', 32, 16]`.

```text
Network: [B, 16, 32] -> [B, 32, 16]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    transpose  x=[B, 16, 32]  out=[B, 32, 16]
```

Parameters: 0

### Example 2

Swaps height and width of an image tensor, leaving the channel axis alone.

```python
transpose(2, 3)
```

Input `['B', 3, 8, 16]` → output `['B', 3, 16, 8]`.

```text
Network: [B, 3, 8, 16] -> [B, 3, 16, 8]  dtype=float32
index  name  operation  input shapes     output shapes
0      n0    transpose  x=[B, 3, 8, 16]  out=[B, 3, 16, 8]
```

Parameters: 0

### Example 3

The swap is bidirectional: the projection width 10 is read back through it.

```python
linear()
transpose(1, 2)
```

Input `['B', 4, 6]` → output `['B', 10, 4]`.

```text
Network: [B, 4, 6] -> [B, 10, 4]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 6]   out=[B, 4, 10]
1      n1    transpose  x=[B, 4, 10]  out=[B, 10, 4]
```

Parameters: 70
