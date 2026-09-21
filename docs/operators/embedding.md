# `embedding`

Look up a learned vector for every integer token id.

**Category:** sequence · **Identity:** `embedding@1`

## Shape

```text
ids[B, T]:int64 -> out[B, T, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `ids` | input | `ids[B, T]:int64` | int64 |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `num_embeddings` (positional) | int | required | >= 1; <= 2147483647 | Vocabulary size; ids must lie in [0, num_embeddings). |
| `dim` (positional) | int | inferred | >= 1; <= 2147483647; binds `D` | Width of each embedding vector. Omit it to infer the width from what follows. |

## Description

A learned lookup table: row ``i`` of ``weight`` is the vector for token
id ``i``.

```text
out[b, t, :] = weight[ids[b, t], :]
```

The input port ``ids`` carries **int64** values, not activations, so the
graph input must be declared with ``input_dtype="int64"`` when
``embedding`` is the first operation:

```python
network("embedding(1000, 32)\nlinear(8)", input_shape=("B", 16),
        output_shape=("B", 16, 8), input_dtype="int64", device="cpu")
```

Ids are positions on the ``T`` axis of a ``[B, T]`` tensor and the result
is the ``[B, T, D]`` sequence those tokens embed to. Every id must satisfy
``0 <= id < num_embeddings``; an out-of-range id is a CUDA-side or
CPU-side indexing fault, not a shape error, because ids are data.

The single parameter is ``weight`` of shape ``[num_embeddings, dim]``,
initialized from the standard normal distribution. It is dense: the whole
table receives a gradient contribution on every step. There is no padding
index and no norm clipping; train and eval behave identically.

## Examples

### Example 1

A vocabulary of 1000 tokens embedded into 32 features, then projected to 8.

```python
embedding(1000, 32)
linear(8)
```

Input `['B', 16]` (`input_dtype="int64"`) → output `['B', 16, 8]`.

```text
Network: [B, 16] -> [B, 16, 8]  dtype=float32  input_dtype=int64
index  name  operation  input shapes   output shapes
0      n0    embedding  ids=[B, 16]    out=[B, 16, 32]
1      n1    linear     x=[B, 16, 32]  out=[B, 16, 8]
```

Parameters: 32,264

### Example 2

The embedding width 5 is inferred from the output contract.

```python
embedding(64)
relu()
```

Input `['B', 8]` (`input_dtype="int64"`) → output `['B', 8, 5]`.

```text
Network: [B, 8] -> [B, 8, 5]  dtype=float32  input_dtype=int64
index  name  operation  input shapes  output shapes
0      n0    embedding  ids=[B, 8]    out=[B, 8, 5]
1      n1    relu       x=[B, 8, 5]   out=[B, 8, 5]
```

Parameters: 320

### Example 3

A token embedding followed by a class token and learned positions.

```python
embedding(128, 16)
cls_token()
pos_embed(32)
linear(4)
```

Input `['B', 6]` (`input_dtype="int64"`) → output `['B', 7, 4]`.

```text
Network: [B, 6] -> [B, 7, 4]  dtype=float32  input_dtype=int64
index  name  operation  input shapes  output shapes
0      n0    embedding  ids=[B, 6]    out=[B, 6, 16]
1      n1    cls_token  x=[B, 6, 16]  out=[B, 7, 16]
2      n2    pos_embed  x=[B, 7, 16]  out=[B, 7, 16]
3      n3    linear     x=[B, 7, 16]  out=[B, 7, 4]
```

Parameters: 2,644
