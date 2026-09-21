# `broadcast_add`

Elementwise sum of two tensors of equal rank, broadcasting size-1 axes.

**Category:** arithmetic · **Identity:** `broadcast_add@1`

## Shape

```text
a, b -> out
```

Relation: `equal rank and batch; per non-batch axis the extents are equal or one of them is 1; out takes the larger extent`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `a` | input | `a` | compute |
| `b` | input | `b` | compute |
| `out` | output | `out` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

``out = a + b`` with numpy-style broadcasting restricted to operands of
**equal rank**. Use this when one operand is a per-sample, per-channel
bias such as ``[B, C, H, W] + [B, C, 1, 1]``; use `add` when the shapes
match exactly, which is the stricter and cheaper contract.

Broadcasting is symmetric: for every non-batch axis the two extents must
either be equal or one of them must be 1, and the output takes the larger
extent. Either operand may hold the size-1 axis, so ``[B, 1, H, W] +
[B, C, 1, 1]`` is accepted and produces ``[B, C, H, W]``. The batch axis
is never broadcast — both operands carry the full batch — and rank is
never padded, so a ``[B, C, H, W]`` map and a ``[B, C]`` vector are a rank
mismatch (``E_CONSTRAINT``); `reshape` the vector to ``[B, C, 1, 1]``
first. Incompatible extents report ``E_CONSTRAINT`` naming the axis.

Shape inference runs in both directions wherever the answer is unique. A
known pair of inputs always fixes the output. Going backward, a known
output plus one known operand fixes the other operand on every axis where
the known operand has extent 1 (the other must supply the full extent) and
on every axis where the output extent is 1 (both operands must be 1). On
an axis where the known operand already matches a larger output extent the
other operand could be either that extent or 1, so nothing is inferred
there and the extent has to come from the operand's own producer.

### Determinism

The backward pass reduces each broadcast axis with a plain ``sum``
(torch's ``sum_to_size``), which is a deterministic tree reduction with no
atomic accumulation — unlike ``index_add``/``scatter_add``, which are the
usual source of run-to-run drift on CUDA. Both the forward and the
backward run cleanly under
``torch.use_deterministic_algorithms(True)``, and repeated backward passes
over identical inputs on CUDA give bit-identical gradients. The operator
has no parameters and behaves identically in train and eval mode; the sum
is computed in the plan compute dtype.

## Examples

### Example 1

A per-sample, per-channel conditioning bias: the pooled [B, 3, 1, 1] projection is added to every spatial position.

```python
saved = x
global_avg_pool()
linear(3)
bias = reshape(3, 1, 1)
broadcast_add(saved, bias)
```

Input `['B', 3, 8, 8]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 3, 8, 8]  dtype=float32
index  name  operation        input shapes                    output shapes
0      n0    global_avg_pool  x=[B, 3, 8, 8]                  out=[B, 3]
1      n1    linear           x=[B, 3]                        out=[B, 3]
2      n2    reshape          x=[B, 3]                        out=[B, 3, 1, 1]
3      n3    broadcast_add    a=[B, 3, 8, 8], b=[B, 3, 1, 1]  out=[B, 3, 8, 8]
```

Parameters: 12

### Example 2

A squeeze-and-excite style shortcut: the 1x1 channel summary is added back to the map.

```python
h = conv(8, kernel_size=3, padding=1)
pooled = adaptive_avg_pool(h, 1)
broadcast_add(h, pooled)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation          input shapes                    output shapes
0      n0    conv               x=[B, 3, 8, 8]                  out=[B, 8, 8, 8]
1      n1    adaptive_avg_pool  x=[B, 8, 8, 8]                  out=[B, 8, 1, 1]
2      n2    broadcast_add      a=[B, 8, 8, 8], b=[B, 8, 1, 1]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 3

Sequences: a [B, 1, D] summary token is added to every position of a [B, T, D] tensor.

```python
h = linear(8)
summary = mean(h, 1, keepdim=True)
broadcast_add(h, summary)
```

Input `['B', 6, 4]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 4] -> [B, 6, 8]  dtype=float32
index  name  operation      input shapes              output shapes
0      n0    linear         x=[B, 6, 4]               out=[B, 6, 8]
1      n1    mean           x=[B, 6, 8]               out=[B, 1, 8]
2      n2    broadcast_add  a=[B, 6, 8], b=[B, 1, 8]  out=[B, 6, 8]
```

Parameters: 40
