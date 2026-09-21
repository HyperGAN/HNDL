# `dropout`

Randomly zero elements during training and rescale the rest.

**Category:** activation · **Identity:** `dropout@1`

## Shape

```text
x[B, ...] -> out[B, ...]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ...]` | compute |
| `out` | output | `out[B, ...]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `p` (positional) | float | `0.1` | >= 0; < 1 | Probability that an individual element is zeroed while training; 0 disables dropout. |

## Description

Elementwise dropout, computed out of place so shared branches are never
mutated.

In **training mode** each element of the input is independently zeroed with
probability ``p``, and the surviving elements are scaled by ``1 / (1 - p)``
so the expected value of the activation is unchanged:

```text
m ~ Bernoulli(1 - p)          (drawn per element)
out = x * m / (1 - p)
```

In **eval mode** (``model.eval()``) dropout is the identity: ``out = x``,
with no mask and no rescaling. ``p = 0.0`` is the identity in both modes.

The mask is elementwise and shape agnostic: the operator accepts ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` and drops every element of the tensor
independently — it does not drop whole channels or whole positions (that
would be a separate ``dropout2d``-style operator). The output shape always
equals the input shape, and the operator holds no parameters or buffers.

### Randomness

Dropout is the one deliberate exception to HNDL's "no hidden forward
randomness" rule: in training mode ``forward`` draws from torch's **global**
RNG, so two calls on the same input return different results unless the RNG
is reseeded between them, and a plan containing ``dropout(p > 0)`` is only
reproducible if you control ``torch.manual_seed`` yourself. Everything the
resolver sees stays deterministic — the drawn mask never affects shapes,
arguments, or the plan digest.

Because of that, the declared examples all use ``p=0.0``: the generic
example harness builds in training mode and compares two forward passes
without reseeding, which any ``p > 0`` would fail by construction. The
operator's own tests cover the statistics and scaling for ``p > 0``.

The mask is drawn and applied in the plan's compute dtype (float32,
float16, or bfloat16); nothing is upcast.

## Examples

### Example 1

Regularizes the hidden features. The examples pin p=0.0 so that they are reproducible; use dropout(0.1) in a real network.

```python
linear(64)
relu()
dropout(0.0)
linear()
```

Input `['B', 32]` → output `['B', 10]`.

```text
Network: [B, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 32]     out=[B, 64]
1      n1    relu       x=[B, 64]     out=[B, 64]
2      n2    dropout    x=[B, 64]     out=[B, 64]
3      n3    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 2,762

### Example 2

Elementwise on images: every element of [B, C, H, W] is dropped independently.

```python
conv(8, kernel_size=3, padding=1)
dropout(0.0)
relu()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    dropout    x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
2      n2    relu       x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 3

On a [B, T, D] sequence each position and feature is dropped independently.

```python
linear(16)
dropout(0.0)
linear()
```

Input `['B', 4, 8]` → output `['B', 4, 3]`.

```text
Network: [B, 4, 8] -> [B, 4, 3]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 8]   out=[B, 4, 16]
1      n1    dropout    x=[B, 4, 16]  out=[B, 4, 16]
2      n2    linear     x=[B, 4, 16]  out=[B, 4, 3]
```

Parameters: 195
