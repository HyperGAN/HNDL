# `max_pool`

Two-dimensional max pooling over [B, C, H, W] images.

**Category:** spatial · **Identity:** `max_pool2d@1`

## Shape

```text
x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]
```

Relation: `H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1, same for W; stride=0 means stride=kernel_size`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H_in, W_in]` | compute |
| `out` | output | `out[B, C, H_out, W_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `kernel_size` (positional) | pair | required | >= 1 | Pooling window height and width; an int applies to both. |
| `stride` | pair | `(0, 0)` | >= 0 | Step between windows on each axis; 0, the default, steps by one whole kernel_size. |
| `padding` | pair | `(0, 0)` | >= 0 | Implicit -inf padding on each spatial side; at most kernel_size//2 per axis. |

## Description

Takes the maximum over each ``kernel_size`` window of the spatial axes
of a ``[B, C, H, W]`` image, stepping by ``stride`` and treating the
``padding`` border as negative infinity so padded positions never win.
Batch and channels pass through unchanged and

``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

with the same rule on width; dilation is 1 and ``ceil_mode`` is false, so
a trailing partial window is dropped. ``stride=0`` is the default and
means "the same as ``kernel_size``", which is PyTorch's own default of
non-overlapping windows; any other value is used as written. ``padding``
may not exceed ``kernel_size//2`` on an axis. There are no parameters and
no train/eval difference. The gradient routes to the argmax of each
window, so ties send it to a single position.

## Examples

### Example 1

A 2x2 window with the default stride halves height and width.

```python
conv(8, kernel_size=3, padding=1)
relu()
max_pool(2)
```

Input `['B', 3, 16, 16]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 16, 16] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    conv       x=[B, 3, 16, 16]  out=[B, 8, 16, 16]
1      n1    relu       x=[B, 8, 16, 16]  out=[B, 8, 16, 16]
2      n2    max_pool   x=[B, 8, 16, 16]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 2

Downsample, then classify the 4*4*4 remaining activations.

```python
max_pool(2)
flatten()
linear()
```

Input `['B', 4, 8, 8]` → output `['B', 10]`.

```text
Network: [B, 4, 8, 8] -> [B, 10]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    max_pool   x=[B, 4, 8, 8]  out=[B, 4, 4, 4]
1      n1    flatten    x=[B, 4, 4, 4]  out=[B, 64]
2      n2    linear     x=[B, 64]       out=[B, 10]
```

Parameters: 650

### Example 3

Overlapping 3x3 windows with stride 2 and padding 1.

```python
max_pool(3, stride=2, padding=1)
```

Input `['B', 3, 15, 15]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 15, 15] -> [B, 3, 8, 8]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    max_pool   x=[B, 3, 15, 15]  out=[B, 3, 8, 8]
```

Parameters: 0
