# `pos_embed`

Add a learned position vector to every position of a sequence.

**Category:** sequence · **Identity:** `pos_embed@1`

## Shape

```text
x[B, T, D] -> out[B, T, D]
```

Relation: `out == x elementwise; T <= max_len`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `max_len` (positional) | int | required | >= 1; <= 2147483647 | Rows in the position table; the longest sequence this layer can encode. |

## Description

A learned absolute position encoding, added to the sequence:

```text
out[b, t, :] = x[b, t, :] + weight[t, :]
```

``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features;
the same position vector is added to every example in the batch. The
parameter ``weight`` has shape ``[max_len, D]`` and is initialized from a
normal distribution with standard deviation 0.02, the convention for
transformer position tables. Only the first ``T`` rows take part in the
forward pass, so only those rows receive a gradient; the remaining rows
are kept so that the same module can encode longer sequences after a
re-resolve.

``max_len`` is a capacity, not a shape: the resolver only requires
``T <= max_len`` and reports ``E_CONSTRAINT`` when a resolved sequence is
longer than the table. Train and eval behave identically, and the addition
is performed in the input dtype.

## Examples

### Example 1

Positions 0..5 of the table are added; rows 6..63 stay unused.

```python
pos_embed(64)
linear(8)
```

Input `['B', 6, 4]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 4] -> [B, 6, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    pos_embed  x=[B, 6, 4]   out=[B, 6, 4]
1      n1    linear     x=[B, 6, 4]   out=[B, 6, 8]
```

Parameters: 296

### Example 2

The usual transformer input stem: token embeddings plus learned positions.

```python
embedding(500, 16)
pos_embed(32)
linear(4)
```

Input `['B', 12]` → output `['B', 12, 4]`. Graph input dtype: `int64`.

```text
Network: [B, 12] -> [B, 12, 4]  dtype=float32  input_dtype=int64
index  name  operation  input shapes   output shapes
0      n0    embedding  ids=[B, 12]    out=[B, 12, 16]
1      n1    pos_embed  x=[B, 12, 16]  out=[B, 12, 16]
2      n2    linear     x=[B, 12, 16]  out=[B, 12, 4]
```

Parameters: 8,580
