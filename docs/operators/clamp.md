# `clamp`

Saturate every element into the closed interval [min, max].

**Category:** arithmetic · **Identity:** `clamp@1`

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
| `min` (positional) | float | required | — | Lower bound; elements below it become min. |
| `max` (positional) | float | required | — | Upper bound; elements above it become max. Must be >= min. |

## Description

Elementwise ``out = min(max(x, min), max)``, computed as
``torch.clamp(x, min=min, max=max)`` out of place.

The operation is shape preserving: every supported rank (``[B, F]``,
``[B, T, D]``, ``[B, C, H, W]``) passes through unchanged, and no axis has
a special meaning. Both bounds are required floats and ``min <= max`` is
checked when arguments are normalized (``E_ARGUMENT`` otherwise).

The gradient is one strictly inside the interval and zero outside it, so a
saturated element stops contributing to its input's gradient; elements
exactly on a bound keep a gradient of one, matching ``torch.clamp``.
Behavior is identical in train and eval mode and there are no parameters.
Computation stays in the input dtype, so ``float16`` and ``bfloat16``
activations are clamped without an upcast.

## Examples

### Example 1

A ReLU6-style activation on the hidden features.

```python
linear(64)
clamp(0.0, 6.0)
linear()
```

Input `['B', 128]` → output `['B', 10]`.

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 128]    out=[B, 64]
1      n1    clamp      x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 8,906

### Example 2

Saturating an image tensor into [-1, 1].

```python
conv(8, kernel_size=3, padding=1)
clamp(-1.0, 1.0)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    clamp      x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 3

On a [B, T, D] sequence the bounds apply to every element.

```python
linear(8)
clamp(0.0, 1.0)
linear()
```

Input `['B', 4, 6]` → output `['B', 4, 3]`.

```text
Network: [B, 4, 6] -> [B, 4, 3]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 6]   out=[B, 4, 8]
1      n1    clamp      x=[B, 4, 8]   out=[B, 4, 8]
2      n2    linear     x=[B, 4, 8]   out=[B, 4, 3]
```

Parameters: 83
