# `cls_token`

Prepend one learned classification token to a sequence.

**Category:** sequence · **Identity:** `cls_token@1`

## Shape

```text
x[B, T, D] -> out[B, T_plus, D]
```

Relation: `T_plus == T + 1; the feature width D is unchanged`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, T_plus, D]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Prepends one learned vector to the position axis of a sequence:

```text
out = cat([token.expand(B, 1, D), x], dim=1)
out[:, 0, :] = token        # the class position
out[:, 1:, :] = x           # the original sequence
```

``x`` is a ``[B, T, D]`` sequence of ``T`` positions with ``D`` features
and the result has ``T + 1`` positions, so every downstream operator that
counts positions — a position table, a pooling step, an output contract —
sees the extra slot. The relation is bidirectional: a known input length
fixes the output length and a known output length fixes the input length.

The single parameter is ``token`` of shape ``[1, 1, D]``, initialized from
a normal distribution with standard deviation 0.02 and broadcast across
the batch, so every example starts from the same learned summary vector.
It is a plain concatenation with no projection and no normalization;
train and eval behave identically.

## Examples

### Example 1

Four positions become five; the projection follows the token.

```python
cls_token()
linear(8)
```

Input `['B', 4, 6]` → output `['B', 5, 8]`.

```text
Network: [B, 4, 6] -> [B, 5, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    cls_token  x=[B, 4, 6]   out=[B, 5, 6]
1      n1    linear     x=[B, 5, 6]   out=[B, 5, 8]
```

Parameters: 62

### Example 2

A token sequence gains a summary position before the projection.

```python
embedding(64, 8)
cls_token()
linear(4)
```

Input `['B', 6]` → output `['B', 7, 4]`. Graph input dtype: `int64`.

```text
Network: [B, 6] -> [B, 7, 4]  dtype=float32  input_dtype=int64
index  name  operation  input shapes  output shapes
0      n0    embedding  ids=[B, 6]    out=[B, 6, 8]
1      n1    cls_token  x=[B, 6, 8]   out=[B, 7, 8]
2      n2    linear     x=[B, 7, 8]   out=[B, 7, 4]
```

Parameters: 556

### Example 3

The class token is counted by the position table that follows it.

```python
linear(12)
cls_token()
pos_embed(16)
linear()
```

Input `['B', 4, 6]` → output `['B', 5, 3]`.

```text
Network: [B, 4, 6] -> [B, 5, 3]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 6]   out=[B, 4, 12]
1      n1    cls_token  x=[B, 4, 12]  out=[B, 5, 12]
2      n2    pos_embed  x=[B, 5, 12]  out=[B, 5, 12]
3      n3    linear     x=[B, 5, 12]  out=[B, 5, 3]
```

Parameters: 327
