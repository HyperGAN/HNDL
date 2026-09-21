# `pixel_shuffle`

Trade channels for resolution: rearrange C*r^2 channels into an r-times larger image.

**Category:** spatial · **Identity:** `pixel_shuffle@1`

## Shape

```text
x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]
```

Relation: `C_in == C_out * upscale_factor**2; H_out = H_in * upscale_factor, W_out = W_in * upscale_factor`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C_in, H_in, W_in]` | compute |
| `out` | output | `out[B, C_out, H_out, W_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `upscale_factor` (positional) | int | required | >= 1 | Factor r: consumes r^2 channels per output channel. |

## Description

Rearranges a ``[B, C*r^2, H, W]`` tensor into ``[B, C, H*r, W*r]``,
where ``r`` is ``upscale_factor``. Channel ``c*r^2 + i*r + j`` of the
input supplies output channel ``c`` at the sub-pixel offset ``(i, j)``
inside each ``r`` x ``r`` output block:

```text
out[b, c, h*r + i, w*r + j] == x[b, c*r^2 + i*r + j, h, w]
```

This is the standard sub-pixel convolution upsampler: a preceding
``conv`` produces ``r^2`` values per output pixel and the shuffle lays
them out spatially, which avoids the checkerboard artifacts of a strided
transposed convolution. It is a pure permutation and reshape, with no
parameters, no arithmetic, and identical train and eval behavior; the
output dtype is the input dtype.

Input channels must be divisible by ``upscale_factor**2`` and a known
output extent must be divisible by ``upscale_factor``; otherwise
``E_CONSTRAINT`` is reported. Resolution runs in both directions, so the
channel width of the producing layer can be left to the solver.

## Examples

### Example 1

The sub-pixel convolution of ESPCN: 48 = 3 * 4^2 channels become a 4x larger RGB image.

```python
conv(48, kernel_size=3, padding=1)
pixel_shuffle(4)
```

Input `['B', 16, 8, 8]` → output `['B', 3, 32, 32]`.

```text
Network: [B, 16, 8, 8] -> [B, 3, 32, 32]  dtype=float32
index  name  operation      input shapes     output shapes
0      n0    conv           x=[B, 16, 8, 8]  out=[B, 48, 8, 8]
1      n1    pixel_shuffle  x=[B, 48, 8, 8]  out=[B, 3, 32, 32]
```

Parameters: 6,960

### Example 2

A bare shuffle: 12 channels at 4x4 become 3 channels at 8x8.

```python
pixel_shuffle(2)
```

Input `['B', 12, 4, 4]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 12, 4, 4] -> [B, 3, 8, 8]  dtype=float32
index  name  operation      input shapes     output shapes
0      n0    pixel_shuffle  x=[B, 12, 4, 4]  out=[B, 3, 8, 8]
```

Parameters: 0

### Example 3

The convolution width 20 is inferred backward from the shuffled output.

```python
conv(kernel_size=3, padding=1)
pixel_shuffle(2)
```

Input `['B', 3, 8, 8]` → output `['B', 5, 16, 16]`.

```text
Network: [B, 3, 8, 8] -> [B, 5, 16, 16]  dtype=float32
index  name  operation      input shapes     output shapes
0      n0    conv           x=[B, 3, 8, 8]   out=[B, 20, 8, 8]
1      n1    pixel_shuffle  x=[B, 20, 8, 8]  out=[B, 5, 16, 16]
```

Parameters: 560
