# `gelu`

Gaussian error linear unit, x * Phi(x).

**Category:** activation · **Identity:** `gelu@1`

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
| `approximate` (positional) | str | `"none"` | one of `none`, `tanh` | Formula to use: "none" for the exact erf form, "tanh" for the tanh approximation. |

## Description

Elementwise ``out = x * Phi(x)``, where ``Phi`` is the standard normal
cumulative distribution function.

With ``approximate="none"`` (the default) the exact form is computed:

```text
out = 0.5 * x * (1 + erf(x / sqrt(2)))
```

With ``approximate="tanh"`` the cheaper tanh approximation is used, which
is what the original BERT and GPT-2 implementations shipped:

```text
out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

The two forms agree to roughly 1e-3 in absolute value but are not
interchangeable when loading weights trained against a specific one, so
the choice is recorded in the plan.

The operator is elementwise and shape preserving on ``[B, F]``,
``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
identically in train and eval mode, and is computed out of place in the
plan's compute dtype.

## Examples

### Example 1

The usual transformer feed-forward activation on [B, F] features.

```python
linear(64)
gelu()
linear()
```

Input `['B', 128]` → output `['B', 10]`.

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 128]    out=[B, 64]
1      n1    gelu       x=[B, 64]     out=[B, 64]
2      n2    linear     x=[B, 64]     out=[B, 10]
```

Parameters: 8,906

### Example 2

On a [B, T, D] sequence the activation applies elementwise at every position.

```python
linear(32)
gelu(approximate="tanh")
linear()
```

Input `['B', 6, 16]` → output `['B', 6, 8]`.

```text
Network: [B, 6, 16] -> [B, 6, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 6, 16]  out=[B, 6, 32]
1      n1    gelu       x=[B, 6, 32]  out=[B, 6, 32]
2      n2    linear     x=[B, 6, 32]  out=[B, 6, 8]
```

Parameters: 808

### Example 3

Elementwise over every channel and spatial position of a [B, C, H, W] tensor.

```python
conv(8, kernel_size=3, padding=1)
gelu()
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    gelu       x=[B, 8, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 224
