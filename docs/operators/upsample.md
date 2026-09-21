# `upsample`

Enlarge height and width by an integer factor with a fixed interpolation kernel.

**Category:** spatial · **Identity:** `upsample@1`

## Shape

```text
x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]
```

Relation: `H_out = H_in * scale_factor, W_out = W_in * scale_factor; channels unchanged`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H_in, W_in]` | compute |
| `out` | output | `out[B, C, H_out, W_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `scale_factor` (positional) | int | `2` | >= 1 | Integer factor applied to height and width. |
| `mode` | str | `"nearest"` | one of `nearest`, `bilinear`, `bicubic` | Interpolation kernel: nearest, bilinear, or bicubic. |
| `align_corners` | bool | `False` | — | Align the corner pixels of input and output; bilinear and bicubic only. |

## Description

Resamples the spatial axes of a ``[B, C, H, W]`` image by an exact
integer factor: ``H_out = H * scale_factor`` and
``W_out = W * scale_factor``. The channel axis is untouched and there are
no parameters, so train and eval behave identically and the layer is a
pure function of its input.

Modes:

- ``nearest`` repeats each input pixel in a ``scale_factor`` x
  ``scale_factor`` block. It is the cheapest choice and is exactly
  equivalent to ``x.repeat_interleave(scale_factor, 2).repeat_interleave(scale_factor, 3)``.
- ``bilinear`` and ``bicubic`` interpolate between neighbouring pixels.
  ``align_corners`` selects the sampling grid convention: ``False`` (the
  default) treats pixels as areas, ``True`` pins the corner pixel centers
  of input and output together.

``align_corners`` is meaningful only for ``bilinear`` and ``bicubic``;
with ``nearest`` it must stay ``False``, and the underlying PyTorch call
receives ``None`` so no warning is emitted. Note that ``bicubic`` can
overshoot the input range; clamp afterwards if bounded output matters.

Resolution is bidirectional: a known input extent fixes the output, and a
known output extent fixes the input when it divides by ``scale_factor``,
otherwise ``E_CONSTRAINT`` is reported.

## Examples

### Example 1

Nearest-neighbour doubling followed by a convolution that smooths the blocks.

```python
upsample(2)
conv(3, kernel_size=3, padding=1)
```

Input `['B', 3, 8, 8]` → output `['B', 3, 16, 16]`.

```text
Network: [B, 3, 8, 8] -> [B, 3, 16, 16]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    upsample   x=[B, 3, 8, 8]    out=[B, 3, 16, 16]
1      n1    conv       x=[B, 3, 16, 16]  out=[B, 3, 16, 16]
```

Parameters: 84

### Example 2

Bilinear doubling with align_corners=False, the PyTorch default.

```python
upsample(2, mode="bilinear")
```

Input `['B', 4, 8, 8]` → output `['B', 4, 16, 16]`.

```text
Network: [B, 4, 8, 8] -> [B, 4, 16, 16]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    upsample   x=[B, 4, 8, 8]  out=[B, 4, 16, 16]
```

Parameters: 0

### Example 3

A single stage that quadruples both spatial axes.

```python
conv(8, kernel_size=3, padding=1)
upsample(4)
```

Input `['B', 3, 4, 4]` → output `['B', 8, 16, 16]`.

```text
Network: [B, 3, 4, 4] -> [B, 8, 16, 16]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 4, 4]  out=[B, 8, 4, 4]
1      n1    upsample   x=[B, 8, 4, 4]  out=[B, 8, 16, 16]
```

Parameters: 224

### Example 4

The 4x4 seed and the projection width 256 are inferred backward through the upsampling.

```python
linear()
reshape(16)
upsample(2)
```

Input `['B', 32]` → output `['B', 16, 8, 8]`.

```text
Network: [B, 32] -> [B, 16, 8, 8]  dtype=float32
index  name  operation  input shapes     output shapes
0      n0    linear     x=[B, 32]        out=[B, 256]
1      n1    reshape    x=[B, 256]       out=[B, 16, 4, 4]
2      n2    upsample   x=[B, 16, 4, 4]  out=[B, 16, 8, 8]
```

Parameters: 8,448
