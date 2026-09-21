# `gather_token`

Select the token of a sequence at the position holding the largest id.

**Category:** sequence · **Identity:** `gather_token@1`

## Shape

```text
x[B, T, D], ids[B, T]:int64 -> out[B, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `ids` | input | `ids[B, T]:int64` | int64 |
| `out` | output | `out[B, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `mode` | str | `"argmax"` | one of `argmax` | How the position is chosen from ids; only argmax is supported today. |

## Description

Picks one position out of a ``[B, T, D]`` sequence per example and
returns the ``[B, D]`` vector found there. Which position is chosen is
decided by a second input, ``ids``, an int64 tensor of shape ``[B, T]``:

```text
p[b]      = argmax_t ids[b, t]
out[b, :] = x[b, p[b], :]
```

This is CLIP's end-of-text pooling. A tokenizer writes the end-of-text
marker as the highest id in the vocabulary, so the argmax of the id row
lands on the final real token of that example and the padding after it is
ignored. Unlike ``pool_tokens("last")``, which always takes position
``T - 1``, the selected position varies per example, which is what a
padded batch needs.

``ids`` is data, not activations. It is typically the graph input itself,
declared with ``input_dtype="int64"`` and fanned out to both
``embedding`` and this operator:

```python
network('tokens = embedding(x, 32, 8)\n'
        'features = linear(tokens, 8)\n'
        'gather_token(features, x)',
        input_shape=("B", 6), output_shape=("B", 8),
        input_dtype="int64", device="cpu")
```

Both inputs are explicit: neither port is filled by the implicit current
tensor, so the configuration names the sequence and the ids. The two must
agree on the batch and on the token count ``T``; a mismatch is a shape
contradiction, reported as ``E_CONSTRAINT`` at resolution.

Ties follow ``torch.argmax``: when a row holds its maximum id more than
once, the first such position wins. An all-equal row therefore selects
position 0. Ids are never bounds-checked against the vocabulary here,
because only their ordering matters.

``mode`` exists so that other selection rules can be added later; today
``"argmax"`` is the only choice.

There are no parameters and no state, so train and eval behave
identically. The gather is exact in every compute dtype: values are moved,
never combined. Gradients flow to ``x`` only, as a scatter that leaves
every unselected position at zero; ``ids`` is integer and carries none.

## Examples

### Example 1

CLIP-style end-of-text pooling: the ids fan out to the embedding and to the gather.

```python
tokens = embedding(x, 32, 8)
features = linear(tokens, 8)
gather_token(features, x)
```

Input `['B', 6]` (`input_dtype="int64"`) → output `['B', 8]`.

```text
Network: [B, 6] -> [B, 8]  dtype=float32  input_dtype=int64
index  name  operation     input shapes             output shapes
0      n0    embedding     ids=[B, 6]               out=[B, 6, 8]
1      n1    linear        x=[B, 6, 8]              out=[B, 6, 8]
2      n2    gather_token  x=[B, 6, 8], ids=[B, 6]  out=[B, 8]
```

Parameters: 328

### Example 2

A small text encoder: the pooled end-of-text feature is projected to the output width.

```python
h = embedding(x, 64, 16)
h = pos_embed(h, 32)
h = transformer_block(h, 4)
pooled = gather_token(h, x)
linear(pooled)
```

Input `['B', 10]` (`input_dtype="int64"`) → output `['B', 12]`.

```text
Network: [B, 10] -> [B, 12]  dtype=float32  input_dtype=int64
index  name  operation          input shapes                output shapes
0      n0    embedding          ids=[B, 10]                 out=[B, 10, 16]
1      n1    pos_embed          x=[B, 10, 16]               out=[B, 10, 16]
2      n2    transformer_block  x=[B, 10, 16]               out=[B, 10, 16]
3      n3    gather_token       x=[B, 10, 16], ids=[B, 10]  out=[B, 16]
4      n4    linear             x=[B, 16]                   out=[B, 12]
```

Parameters: 5,020
