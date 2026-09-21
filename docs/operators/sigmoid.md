# `sigmoid`

Logistic sigmoid, squashing values into (0, 1).

**Category:** activation · **Identity:** `sigmoid@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Elementwise ``out = 1 / (1 + exp(-x))``.

The output lies strictly in ``(0, 1)``, which makes the operator a natural
final activation for probabilities and for images normalized to the unit
interval. Gradients vanish for inputs far from zero, so it is a poor
choice for hidden layers; prefer `silu` or `gelu` there. When the loss is
a binary cross entropy, keep the logits and use a fused loss rather than
stacking `sigmoid` in front of it.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.

## Examples

### Example 1

A binary classification head emitting a probability.

```python
linear(1)
sigmoid()
```

Input `['B', 32]` → output `['B', 1]`.

```text
Network: [B, 32] -> [B, 1]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 1]
1      n1    sigmoid    x=[B, 1]      out=[B, 1]
```

Parameters: 33

### Example 2

On a [B, T, D] sequence the activation applies elementwise at every position.

```python
linear(32)
sigmoid()
linear()
```

Input `['B', 6, 16]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 16] -> [B, 6, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 6, 16]  out=[B, 6, 32]
1      n1    sigmoid    x=[B, 6, 32]  out=[B, 6, 32]
2      n2    linear     x=[B, 6, 32]  out=[B, 6, 8]
```

Parameters: 808

### Example 3

A final activation for image generators that emit values in [0, 1].

```python
conv(3, kernel_size=3, padding=1)
sigmoid()
```

Input `['B', 8, 16, 16]` → output `['B', 3, 16, 16]`.

```text
Network: [B, 8, 16, 16] -> [B, 3, 16, 16]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    conv       x=[B, 8, 16, 16]  out=[B, 3, 16, 16]
1      n1    sigmoid    x=[B, 3, 16, 16]  out=[B, 3, 16, 16]
```

Parameters: 219
