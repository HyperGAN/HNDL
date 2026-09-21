# `scale`

Multiply a tensor by a fixed scalar.

**Category:** arithmetic · **Identity:** `scale@1`

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
| `factor` (positional) | float | required | — | Constant every element is multiplied by; it is not learned. |

## Description

``out = factor * x``: a shape-, rank-, and dtype-preserving constant
gain with no parameters and no buffers. ``factor`` is a plan constant, not
a learned value, so the gradient is ``factor * grad_out`` and the behavior
is identical in train and eval mode. The multiplication happens in the
input dtype; a large ``factor`` can overflow float16, so keep the product
inside the representable range for the plan's compute dtype.

## Examples

### Example 1

A residual whose branch is damped by 0.5 before the sum.

```python
saved = x
linear(8, name="branch")
relu()
h = linear(4)
s = scale(h, 0.5)
add(s, saved)
```

Input `['B', 4]` → output `['B', 4]`.

```text
Network: [B, 4] -> [B, 4]  dtype=float32
index  name    operation  input shapes        output shapes
0      branch  linear     x=[B, 4]            out=[B, 8]
1      n1      relu       x=[B, 8]            out=[B, 8]
2      n2      linear     x=[B, 8]            out=[B, 4]
3      n3      scale      x=[B, 4]            out=[B, 4]
4      n4      add        a=[B, 4], b=[B, 4]  out=[B, 4]
```

Parameters: 76

### Example 2

Sequences: the factor applies to every position and feature.

```python
linear(8)
scale(2.0)
tanh()
```

Input `['B', 5, 4]` → output `['B', 5, 8]`.

```text
Network: [B, 5, 4] -> [B, 5, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 5, 4]   out=[B, 5, 8]
1      n1    scale      x=[B, 5, 8]   out=[B, 5, 8]
2      n2    tanh       x=[B, 5, 8]   out=[B, 5, 8]
```

Parameters: 40

### Example 3

Images: a constant gain on the convolution output.

```python
conv(4, kernel_size=3, padding=1)
scale(0.1)
relu()
```

Input `['B', 3, 8, 8]` → output `['B', 4, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 4, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 4, 8, 8]
1      n1    scale      x=[B, 4, 8, 8]  out=[B, 4, 8, 8]
2      n2    relu       x=[B, 4, 8, 8]  out=[B, 4, 8, 8]
```

Parameters: 112
