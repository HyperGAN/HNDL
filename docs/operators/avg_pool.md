# `avg_pool`

Two-dimensional average pooling over [B, C, H, W] images.

**Category:** spatial · **Identity:** `avg_pool2d@1`

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
| `padding` | pair | `(0, 0)` | >= 0 | Implicit zero padding on each spatial side; at most kernel_size//2 per axis. |
| `count_include_pad` | bool | `True` | — | Divide padded windows by the full window area instead of their real elements. |

## Description

Averages each ``kernel_size`` window of the spatial axes of a
``[B, C, H, W]`` image, stepping by ``stride``. Batch and channels pass
through unchanged and

``H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1``

with the same rule on width; ``ceil_mode`` is false, so a trailing
partial window is dropped. ``stride=0`` is the default and means "the
same as ``kernel_size``", which is PyTorch's own default of
non-overlapping windows; any other value is used as written. ``padding``
adds zeros and may not exceed ``kernel_size//2`` on an axis: with
``count_include_pad`` true, the default, those zeros count in the
denominator, and with it false each window divides by the number of real
input elements it covers. There are no parameters and no train/eval
difference; the gradient is spread equally over every counted position.
The sum accumulates in the input dtype, so a large window in float16
loses precision.

## Examples

### Example 1

A 2x2 window with the default stride halves height and width.

```python
conv(8, kernel_size=3, padding=1)
relu()
avg_pool(2)
```

Input `['B', 3, 16, 16]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 16, 16] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    conv       x=[B, 3, 16, 16]  out=[B, 8, 16, 16]
1      n1    relu       x=[B, 8, 16, 16]  out=[B, 8, 16, 16]
2      n2    avg_pool   x=[B, 8, 16, 16]  out=[B, 8, 8, 8]
```

Parameters: 224

### Example 2

A window covering the whole image is global average pooling.

```python
avg_pool(4)
flatten()
linear()
```

Input `['B', 8, 4, 4]` → output `['B', 5]`.

```text
Network: [B, 8, 4, 4] -> [B, 5]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    avg_pool   x=[B, 8, 4, 4]  out=[B, 8, 1, 1]
1      n1    flatten    x=[B, 8, 1, 1]  out=[B, 8]
2      n2    linear     x=[B, 8]        out=[B, 5]
```

Parameters: 45

### Example 3

Overlapping windows where the border averages only real elements.

```python
avg_pool(3, stride=2, padding=1, count_include_pad=False)
```

Input `['B', 3, 15, 15]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 15, 15] -> [B, 3, 8, 8]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    avg_pool   x=[B, 3, 15, 15]  out=[B, 3, 8, 8]
```

Parameters: 0
