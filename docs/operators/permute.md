# `permute`

Reorder the non-batch axes, keeping the batch axis first.

**Category:** shape · **Identity:** `permute@1`

## Shape

```text
x -> out
```

Relation: `out[i] == x[dims[i - 1]] for every non-batch axis i; rank == len(dims) + 1`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dims` (positional) | ints | `()` | >= 1; <= 3 | The non-batch axes 1..rank-1 of the input in their new order, given positionally: permute(3, 1, 2) sends axis 3 to position 1. The batch axis 0 is implicit and cannot be listed. The length fixes the rank. |

Positional values fill `dims`.

## Description

Returns ``x.permute(0, *dims)``. ``dims`` lists the non-batch axes of
the input in the order they take in the output, so ``out`` axis ``i``
carries ``x`` axis ``dims[i - 1]``: ``permute(3, 1, 2)`` turns
``[B, C, H, W]`` into ``[B, W, C, H]``. The batch axis is never listed and
always stays first.

``dims`` must be a permutation of ``1..rank-1``, which means its length
alone fixes the rank of both ports: ``permute(2, 1)`` is rank 3 and
``permute(2, 3, 1)`` is rank 4. A list that is not a permutation of that
range, one that includes the batch axis 0, or one whose length disagrees
with a rank the neighbouring operations already fixed, is reported as
``E_ARGUMENT``. Rank 4 is ``[B, C, H, W]``; rank 3 is ``[B, T, D]`` for
the sequence operations and ``[B, C, L]`` for 1-D convolution, and
``permute(2, 1)`` bridges those two readings.

The result is a view sharing storage and autograd history with the input,
generally non-contiguous; ``reshape`` and ``flatten`` copy when they must.
There are no parameters and no train/eval difference, and the stride
permutation is exact in every compute dtype.

## Examples

### Example 1

A rank-3 swap: the [B, T, D] sequence becomes the [B, C, L] layout.

```python
permute(2, 1)
```

Input `['B', 16, 32]` → output `['B', 32, 16]`.

```text
Network: [B, 16, 32] -> [B, 32, 16]  dtype=float32
index  name  operation  input shapes   output shapes
0      n0    permute    x=[B, 16, 32]  out=[B, 32, 16]
```

Parameters: 0

### Example 2

Width moves to the front of the non-batch axes; channels and height follow.

```python
permute(3, 1, 2)
```

Input `['B', 3, 8, 16]` → output `['B', 16, 3, 8]`.

```text
Network: [B, 3, 8, 16] -> [B, 16, 3, 8]  dtype=float32
index  name  operation  input shapes     output shapes
0      n0    permute    x=[B, 3, 8, 16]  out=[B, 16, 3, 8]
```

Parameters: 0

### Example 3

[B, C, H, W] to a channels-last [B, H, W, C] reading of the same data.

```python
conv(4, kernel_size=3, padding=1)
permute(2, 3, 1)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 4]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 4]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 4, 8, 8]
1      n1    permute    x=[B, 4, 8, 8]  out=[B, 8, 8, 4]
```

Parameters: 112

### Example 4

Bidirectional: the projection width 10 is inferred backward through the reorder.

```python
linear()
permute(2, 1)
```

Input `['B', 4, 6]` → output `['B', 10, 4]`.

```text
Network: [B, 4, 6] -> [B, 10, 4]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 6]   out=[B, 4, 10]
1      n1    permute    x=[B, 4, 10]  out=[B, 10, 4]
```

Parameters: 70
