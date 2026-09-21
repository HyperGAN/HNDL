# `pool_tokens`

Reduce a [B, T, D] sequence to one [B, D] vector per example.

**Category:** sequence · **Identity:** `pool_tokens@1`

## Shape

```text
x[B, T, D] -> out[B, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `mode` (positional) | str | `"mean"` | one of `mean`, `first`, `last`, `max` | Reduction over the token axis: mean, first, last, or max. |

## Description

Collapses the token axis of a rank-3 sequence ``[B, T, D]`` into a
single vector per example, giving ``[B, D]``. ``T`` is the position axis
and ``D`` the feature axis; the feature axis is carried through unchanged,
so the pooled width always equals the incoming width.

```text
mean   out[b, d] = (1 / T) * sum_t x[b, t, d]
first  out[b, d] = x[b, 0, d]
last   out[b, d] = x[b, T - 1, d]
max    out[b, d] = max_t x[b, t, d]
```

``mean`` averages every position, which is the default and the usual
choice for bidirectional encoders. ``first`` picks the leading token, the
``[CLS]`` convention. ``last`` picks the final token, the convention for
causal models whose sequences all end at the same position; it is the
wrong choice when the useful position varies per example. ``max`` takes
the elementwise maximum, resolving ties to the same value rather than to
a position.

There are no parameters and no state, so train and eval behave
identically. The reduction runs in the incoming dtype: with ``mean`` in
float16 a long sequence accumulates rounding error in the sum, which is
inherent to computing in the plan's compute dtype.

## Examples

### Example 1

Averages the 6 token vectors, then classifies the pooled feature.

```python
linear(16)
pool_tokens()
linear()
```

Input `['B', 6, 8]` → output `['B', 10]`.

```text
Network: [B, 6, 8] -> [B, 10]  dtype=float32
index  name  operation    input shapes  output shapes
0      n0    linear       x=[B, 6, 8]   out=[B, 6, 16]
1      n1    pool_tokens  x=[B, 6, 16]  out=[B, 16]
2      n2    linear       x=[B, 16]     out=[B, 10]
```

Parameters: 314

### Example 2

Takes the final token, the usual pooling for a causal sequence model.

```python
pool_tokens("last")
linear()
```

Input `['B', 4, 32]` → output `['B', 8]`.

```text
Network: [B, 4, 32] -> [B, 8]  dtype=float32
index  name  operation    input shapes  output shapes
0      n0    pool_tokens  x=[B, 4, 32]  out=[B, 32]
1      n1    linear       x=[B, 32]     out=[B, 8]
```

Parameters: 264

### Example 3

Elementwise maximum over the tokens.

```python
linear(12)
relu()
pool_tokens(mode="max")
linear()
```

Input `['B', 5, 6]` → output `['B', 3]`.

```text
Network: [B, 5, 6] -> [B, 3]  dtype=float32
index  name  operation    input shapes  output shapes
0      n0    linear       x=[B, 5, 6]   out=[B, 5, 12]
1      n1    relu         x=[B, 5, 12]  out=[B, 5, 12]
2      n2    pool_tokens  x=[B, 5, 12]  out=[B, 12]
3      n3    linear       x=[B, 12]     out=[B, 3]
```

Parameters: 123

### Example 4

A leading [CLS]-style token is selected after the sequence is projected.

```python
reshape(4)
linear(8)
pool_tokens("first")
```

Input `['B', 64]` → output `['B', 8]`.

```text
Network: [B, 64] -> [B, 8]  dtype=float32
index  name  operation    input shapes  output shapes
0      n0    reshape      x=[B, 64]     out=[B, 4, 16]
1      n1    linear       x=[B, 4, 16]  out=[B, 4, 8]
2      n2    pool_tokens  x=[B, 4, 8]   out=[B, 8]
```

Parameters: 136
