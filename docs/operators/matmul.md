# `matmul`

Batched matrix product of two rank-3 tensors.

**Category:** arithmetic · **Identity:** `matmul@1`

## Shape

```text
a[B, N, K], b[B, K, M] -> out[B, N, M]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `a` | input | `a[B, N, K]` | compute |
| `b` | input | `b[B, K, M]` | compute |
| `out` | output | `out[B, N, M]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

``out[i] = a[i] @ b[i]`` for every batch element ``i``: the operation
``torch.bmm`` performs (``torch.matmul`` agrees at rank 3).

```text
a[B, N, K] @ b[B, K, M] -> out[B, N, M]
out[i, n, m] = sum_k a[i, n, k] * b[i, k, m]
```

Both tensor inputs must be supplied explicitly, as with ``mul`` and
``add``; there is no implicit current-tensor operand for the second port.
The inner dimension ``K`` is a shared symbol, so a known width on either
side fixes the other and a mismatch is reported as ``E_CONSTRAINT`` while
the plan resolves. The batch is shared: there is no broadcasting of the
batch axis or of a missing axis.

Only rank 3 is supported. That is the layout attention works in —
``[B, T, D]`` sequences, or an image feature map flattened to
``[B, C, H*W]`` — and it keeps the operation unambiguous. Reshape first if
a tensor arrives at another rank: ``reshape(C)`` turns ``[B, C, H, W]``
into ``[B, C, H*W]``, and ``reshape(C, H, W)`` turns it back. A rank-2 or
rank-4 tensor on any port fails with ``E_CONSTRAINT`` rather than being
silently folded or broadcast.

There are no parameters and no buffers, train and eval behave identically,
and the product runs in the plan compute dtype. Reduced-precision matrix
multiplies accumulate in float32 on CUDA tensor cores, so float16 and
bfloat16 results are close to, but not bitwise equal to, the same product
done in full precision.

## Examples

### Example 1

The energy map of self-attention: a sequence times its own transpose.

```python
k = transpose(1, 2)
matmul(x, k)
```

Input `['B', 6, 4]` → output `['B', 6, 6]`.

```text
Network: [B, 6, 4] -> [B, 6, 6]  dtype=float32
index  name  operation  input shapes              output shapes
0      n0    transpose  x=[B, 6, 4]               out=[B, 4, 6]
1      n1    matmul     a=[B, 6, 4], b=[B, 4, 6]  out=[B, 6, 6]
```

Parameters: 0

### Example 2

Dot-product attention without projections: scores, softmax, then a weighted sum of x.

```python
k = transpose(1, 2)
e = matmul(x, k)
a = softmax(e, -1)
matmul(a, x)
```

Input `['B', 6, 4]` → output `['B', 6, 4]`.

```text
Network: [B, 6, 4] -> [B, 6, 4]  dtype=float32
index  name  operation  input shapes              output shapes
0      n0    transpose  x=[B, 6, 4]               out=[B, 4, 6]
1      n1    matmul     a=[B, 6, 4], b=[B, 4, 6]  out=[B, 6, 6]
2      n2    softmax    x=[B, 6, 6]               out=[B, 6, 6]
3      n3    matmul     a=[B, 6, 6], b=[B, 6, 4]  out=[B, 6, 4]
```

Parameters: 0

### Example 3

The shared inner dimension K flows backward: the projection width is inferred as 6.

```python
g = linear()
t = transpose(x, 1, 2)
matmul(g, t)
```

Input `['B', 4, 6]` → output `['B', 4, 4]`.

```text
Network: [B, 4, 6] -> [B, 4, 4]  dtype=float32
index  name  operation  input shapes              output shapes
0      n0    linear     x=[B, 4, 6]               out=[B, 4, 6]
1      n1    transpose  x=[B, 4, 6]               out=[B, 6, 4]
2      n2    matmul     a=[B, 4, 6], b=[B, 6, 4]  out=[B, 4, 4]
```

Parameters: 42
