# `grid_sample`

Sample an image using a per-example normalized coordinate grid.

**Category:** spatial · **Identity:** `grid_sample@1`

## Shape

```text
x[B, C, H, W], grid[B, OH, OW, 2] -> out[B, C, OH, OW]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H, W]` | compute |
| `grid` | input | `grid[B, OH, OW, 2]` | compute |
| `out` | output | `out[B, C, OH, OW]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `mode` (positional) | str | `"bilinear"` | one of `bilinear`, `nearest`, `bicubic` | Interpolation kernel. |
| `padding_mode` (positional) | str | `"zeros"` | one of `zeros`, `border`, `reflection` | How samples outside the input are filled. |
| `align_corners` (positional) | bool | `False` | — | Whether -1 and 1 refer to corner pixel centers. |

## Description

A direct ``torch.nn.functional.grid_sample`` with an explicit grid.

The grid's last axis is (x,y) in normalized [-1,1] coordinates. Image
and grid batches must match. Defaults are bilinear interpolation, zero
padding, and align_corners=False. No parameters or random draws;
train and eval are identical. Gradients flow to image and coordinates
as supported by PyTorch. CUDA backward can be nondeterministic, and
PyTorch's sampler does not support double backward.

## Examples

### Example 1

Sample a 4x4 image on an 8x8 grid.

```python
g = coordinate_grid(x, 8, 8)
grid_sample(x, g)
```

Input `['B', 3, 4, 4]` → output `['B', 3, 8, 8]`.

```text
Network: [B, 3, 4, 4] -> [B, 3, 8, 8]  dtype=float32
index  name  operation        input shapes                       output shapes
0      n0    coordinate_grid  x=[B, 3, 4, 4]                     out=[B, 8, 8, 2]
1      n1    grid_sample      x=[B, 3, 4, 4], grid=[B, 8, 8, 2]  out=[B, 3, 8, 8]
```

Parameters: 0
