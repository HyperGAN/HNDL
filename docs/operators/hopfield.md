# `hopfield`

Retrieve learned patterns by iterating the modern Hopfield update on the last axis.

**Category:** memory · **Identity:** `hopfield@1`

## Shape

```text
x[B, ..., D] -> out[B, ..., D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ..., D]` | compute |
| `out` | output | `out[B, ..., D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `patterns` (positional) | int | required | >= 1; <= 2147483647 | Number of stored patterns to learn; the rows of the memory matrix. |
| `beta` | float | `1.0` | > 0 | Inverse temperature of the softmax; larger values retrieve a single pattern more sharply. |
| `steps` | int | `1` | >= 1; <= 1024 | Number of retrieval updates applied in sequence; one update equals attention. |
| `normalize` | bool | `True` | — | Layer-normalize the query over the last axis before the first update. |

## Description

A modern (continuous) Hopfield layer, following Ramsauer et al.,
*Hopfield Networks is All You Need* (2020).

The layer owns a learned memory matrix ``stored`` of shape
``[patterns, D]`` whose rows are the stored patterns ``X``. For a query
``xi`` of width ``D`` it minimizes the energy

```text
E(xi) = -lse(beta, X xi) + 0.5 * xi.T @ xi + log(patterns)/beta + 0.5 * M**2
```

where ``lse`` is the log-sum-exp over the stored patterns and ``M`` is the
largest pattern norm. The concave-convex procedure gives the update that
this layer applies, repeated ``steps`` times:

```text
xi <- X.T @ softmax(beta * X @ xi)
```

Written as a batch of rows, ``xi <- softmax(beta * xi @ stored.T) @
stored``. A single update is exactly dot-product attention with the query
``xi``, the stored patterns as both keys and values, and ``beta`` in place
of the ``1/sqrt(d)`` scale: this layer is attention whose keys and values
are parameters rather than activations. Iterating the update is the
associative-memory reading of the same equation, and the fixed points are
the stored patterns (or, at small ``beta``, their metastable averages).

Retrieval behavior follows ``beta``. With ``beta`` large and the patterns
well separated, a query near a stored pattern makes one softmax weight
dominate, so the output is that pattern and further steps do not move it.
With ``beta`` small the softmax is flat and the output tends toward the
mean of the stored patterns; intermediate values retrieve averages of the
patterns a query is close to. Because the energy decreases monotonically
the iteration converges, in practice within a handful of updates, so
``steps > 1`` mainly sharpens retrieval rather than changing its nature.

Axis convention: the layer acts on the last axis of ``[B, D]`` or
``[B, T, D]`` inputs, so every position of a sequence queries the same
memory independently. Image tensors are rejected; flatten or reshape
first. The output has the shape of the input.

When ``normalize`` is true the query is layer-normalized over the last
axis, without a learned affine and with ``eps = 1e-5``, before the first
update only; later updates consume the previous retrieval unchanged. The
normalization keeps ``beta * X @ xi`` in a usable range when the incoming
activations have drifted in scale.

The only parameter is ``stored``, initialized from ``normal(0, 1/sqrt(D))``
so that initial scores have unit scale. Behavior is identical in train and
eval mode; there are no buffers and no running statistics. Everything is
computed in the input dtype, except that the layer normalization follows
``torch.nn.functional.layer_norm``, which accumulates its statistics in
float32 for float16 and bfloat16 inputs.

## Examples

### Example 1

Sixteen stored patterns of width 32; the memory sits between two projections.

```python
linear(32)
hopfield(16)
linear()
```

Input `['B', 8]` → output `['B', 4]`.

```text
Network: [B, 8] -> [B, 4]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 8]      out=[B, 32]
1      n1    hopfield   x=[B, 32]     out=[B, 32]
2      n2    linear     x=[B, 32]     out=[B, 4]
```

Parameters: 932

### Example 2

A sharp memory iterated three times retrieves close to a single stored pattern.

```python
hopfield(8, beta=4.0, steps=3)
```

Input `['B', 12]` → output `['B', 12]`.

```text
Network: [B, 12] -> [B, 12]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    hopfield   x=[B, 12]     out=[B, 12]
```

Parameters: 96

### Example 3

On a [B, T, D] sequence every position queries the memory independently.

```python
linear(24)
hopfield(12, normalize=False)
linear()
```

Input `['B', 6, 10]` → output `['B', 6, 5]`.

```text
Network: [B, 6, 10] -> [B, 6, 5]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 6, 10]  out=[B, 6, 24]
1      n1    hopfield   x=[B, 6, 24]  out=[B, 6, 24]
2      n2    linear     x=[B, 6, 24]  out=[B, 6, 5]
```

Parameters: 677

### Example 4

A soft memory mixes all four patterns and behaves like a bottleneck.

```python
linear(16)
hopfield(4, beta=0.5)
tanh()
linear()
```

Input `['B', 20]` → output `['B', 3]`.

```text
Network: [B, 20] -> [B, 3]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 20]     out=[B, 16]
1      n1    hopfield   x=[B, 16]     out=[B, 16]
2      n2    tanh       x=[B, 16]     out=[B, 16]
3      n3    linear     x=[B, 16]     out=[B, 3]
```

Parameters: 451
