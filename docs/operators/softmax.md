# `softmax`

Normalize one non-batch axis into a probability distribution.

**Category:** activation · **Identity:** `softmax@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

Relation: `out has the shape of x; dim must resolve to a non-batch axis`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dim` (positional) | int | `-1` | — | Axis to normalize. Negative values count from the end; the resolved axis must not be the batch axis 0. |

## Description

Softmax along one axis:

```
out[..., i, ...] = exp(x[..., i, ...] - m) / sum_j exp(x[..., j, ...] - m)
```

where `m` is the maximum over the axis selected by `dim` and the sum runs
over that same axis (subtracting the maximum is what `torch.softmax` does
internally for numerical stability). Every slice along the axis sums to 1
and every element lies in (0, 1). The shape is preserved.

`dim` indexes the full tensor, batch axis included, following the usual
PyTorch convention: `-1` is the last axis, `1` is the channel axis `C` of a
`[B, C, H, W]` image or the position axis `T` of a `[B, T, D]` sequence,
and `2` is `H`. Normalizing across the batch axis would make examples in a
batch depend on each other, so a `dim` that resolves to axis 0, or that
lies outside the rank of the incoming tensor, is rejected with
`E_ARGUMENT` while the plan resolves.

The computation runs in the activation dtype, including float16 and
bfloat16. Behavior is identical in train and eval mode, and the operator
has no parameters.

## Examples

### Example 1

Class probabilities over the feature axis of a rank-2 tensor.

```python
linear(10)
softmax()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 10]
1      n1    softmax    x=[B, 10]     out=[B, 10]
```

Parameters: 330

### Example 2

A [B, T, D] sequence normalized over its feature axis D.

```python
linear(16)
softmax(-1)
```

Input `['B', 4, 8]` → output `['B', 4, 16]`.

```text
Network: [B, 4, 8] -> [B, 4, 16]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 8]   out=[B, 4, 16]
1      n1    softmax    x=[B, 4, 16]  out=[B, 4, 16]
```

Parameters: 144

### Example 3

Per-pixel distribution across the channel axis of a [B, C, H, W] image.

```python
conv(4, kernel_size=1)
softmax(1)
```

Input `['B', 3, 4, 4]` → output `['B', 4, 4, 4]`.

```text
Network: [B, 3, 4, 4] -> [B, 4, 4, 4]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 4, 4]  out=[B, 4, 4, 4]
1      n1    softmax    x=[B, 4, 4, 4]  out=[B, 4, 4, 4]
```

Parameters: 16
