# `learned_scale`

Multiply a tensor by one learned scalar.

**Category:** arithmetic · **Identity:** `learned_scale@1`

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
| `init_value` | float | `0.0` | — | Value the scalar gamma starts at; 0 makes the layer start as a zero map, the SAGAN convention for a gated residual branch. |

## Description

``out = gamma * x`` where ``gamma`` is a single learned scalar shared by
every element of the tensor:

```text
out[i, ...] = gamma * x[i, ...]
```

The one parameter is named ``gamma`` and has shape ``[1]``, so it is the
target of construction overrides by that exact name:
``learned_scale(init={"gamma": 1.0})`` sets it and
``learned_scale(trainable={"gamma": False})`` freezes it. ``init_value``
gives the value the constructor starts from when no initialization
override applies; it is a float and does not take part in shape inference.

The default ``init_value=0.0`` is the Self-Attention GAN convention (Zhang
et al. 2018): the layer starts as a zero map, so a residual
``add(learned_scale(branch), saved)`` begins exactly as the identity and
the network learns how much of the branch to admit. Note that a gamma of
zero also zeroes the gradient flowing back through ``x``; the branch
parameters start learning through ``gamma`` itself, whose gradient is the
inner product of the branch output with the incoming gradient.

Rank, shape, and dtype are preserved, there are no buffers, and train and
eval behave identically. Use ``scale`` instead for a fixed constant that is
not learned.

## Examples

### Example 1

The SAGAN gate: gamma starts at 0, so the branch begins as the identity and learns its weight.

```python
saved = x
conv(4, kernel_size=3, padding=1)
relu()
h = conv(4, kernel_size=1)
s = learned_scale(h)
add(s, saved)
```

Input `['B', 4, 8, 8]` → output `['B', 4, 8, 8]`.

```text
Network: [B, 4, 8, 8] -> [B, 4, 8, 8]  dtype=float32
index  name  operation      input shapes                    output shapes
0      n0    conv           x=[B, 4, 8, 8]                  out=[B, 4, 8, 8]
1      n1    relu           x=[B, 4, 8, 8]                  out=[B, 4, 8, 8]
2      n2    conv           x=[B, 4, 8, 8]                  out=[B, 4, 8, 8]
3      n3    learned_scale  x=[B, 4, 8, 8]                  out=[B, 4, 8, 8]
4      n4    add            a=[B, 4, 8, 8], b=[B, 4, 8, 8]  out=[B, 4, 8, 8]
```

Parameters: 169

### Example 2

Sequences: one scalar gain, shared by every position and feature, starting at 1.

```python
linear(8)
learned_scale(init_value=1.0)
tanh()
```

Input `['B', 5, 4]` → output `['B', 5, 8]`.

```text
Network: [B, 5, 4] -> [B, 5, 8]  dtype=float32
index  name  operation      input shapes  output shapes
0      n0    linear         x=[B, 5, 4]   out=[B, 5, 8]
1      n1    learned_scale  x=[B, 5, 8]   out=[B, 5, 8]
2      n2    tanh           x=[B, 5, 8]   out=[B, 5, 8]
```

Parameters: 41

### Example 3

A learned gain on a projection; the width 10 still flows back through it.

```python
linear()
learned_scale(init_value=0.5)
```

Input `['B', 16]` → output `['B', 10]`.

```text
Network: [B, 16] -> [B, 10]  dtype=float32
index  name  operation      input shapes  output shapes
0      n0    linear         x=[B, 16]     out=[B, 10]
1      n1    learned_scale  x=[B, 10]     out=[B, 10]
```

Parameters: 171
