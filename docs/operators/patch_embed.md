# `patch_embed`

Cut an image into non-overlapping patches and embed each one as a token.

**Category:** vision · **Identity:** `patch_embed@1`

## Shape

```text
x[B, C, H, W] -> out[B, N, D]
```

Relation: `N == (H / patch_size) * (W / patch_size); patch_size must divide H and W; D == dim`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H, W]` | compute |
| `out` | output | `out[B, N, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dim` (positional) | int | inferred | >= 1; <= 2147483647; binds `D` | Token embedding width. Omit it to infer the width from what follows. |
| `patch_size` (positional) | int | required | >= 1; <= 2147483647 | Side of the square patch; it must divide both the height and the width. |
| `in_channels` | int | inferred | >= 1; <= 2147483647; binds `C` | Image channels at axis 1. Normally inferred from the incoming tensor. |

## Description

The patch-embedding stem of a vision transformer.

The input image ``[B, C, H, W]`` is cut into non-overlapping ``patch_size``
x ``patch_size`` tiles and each tile is projected to ``dim`` features:

```
out = proj(x).flatten(2).transpose(1, 2)
```

where ``proj`` is an ``nn.Conv2d(in_channels, dim, kernel_size=patch_size,
stride=patch_size)``. Because the kernel and the stride are equal, the
convolution reads every pixel exactly once, so the layer is a per-patch
affine map on the flattened ``[C, patch_size, patch_size]`` window.

Axis conventions: the input is an image ``[B, C, H, W]`` and the output is
a sequence ``[B, N, D]`` with ``N = (H / patch_size) * (W / patch_size)``
tokens in row-major order (all patches of the first patch row first) and
``D = dim`` features on the last axis, which is what the sequence
operations such as ``linear`` act on. ``patch_size`` must divide both ``H``
and ``W``; a remainder is reported as ``E_CONSTRAINT`` rather than being
cropped away. Inference runs in both directions: ``dim`` and
``in_channels`` follow the neighbouring contracts, and a known token count
fixes the missing spatial extent when the other one is known. A token
count alone does not determine ``H`` and ``W`` and is reported as
ambiguous instead of guessed.

No positional embedding, class token, or normalization is added; compose
those explicitly. Behavior is identical in train and eval mode. The single
submodule is ``proj``, whose parameters are ``proj.weight`` of shape
``[dim, in_channels, patch_size, patch_size]`` and ``proj.bias``. The
projection runs entirely in the plan compute dtype.

## Examples

### Example 1

A 16x16 image becomes 4x4 = 16 tokens of width 32.

```python
patch_embed(32, 4)
```

Input `['B', 3, 16, 16]` → output `['B', 16, 32]`.

```text
Network: [B, 3, 16, 16] -> [B, 16, 32]  dtype=float32
index  name  operation    input shapes      output shapes
0      n0    patch_embed  x=[B, 3, 16, 16]  out=[B, 16, 32]
```

Parameters: 1,568

### Example 2

Four patch tokens feed an ordinary linear map over the token axis.

```python
patch_embed(16, 8)
linear()
```

Input `['B', 3, 16, 16]` → output `['B', 4, 8]`.

```text
Network: [B, 3, 16, 16] -> [B, 4, 8]  dtype=float32
index  name  operation    input shapes      output shapes
0      n0    patch_embed  x=[B, 3, 16, 16]  out=[B, 4, 16]
1      n1    linear       x=[B, 4, 16]      out=[B, 4, 8]
```

Parameters: 3,224

### Example 3

The embedding width is inferred backward from the output contract.

```python
patch_embed(patch_size=2)
relu()
```

Input `['B', 3, 8, 8]` → output `['B', 16, 64]`.

```text
Network: [B, 3, 8, 8] -> [B, 16, 64]  dtype=float32
index  name  operation    input shapes    output shapes
0      n0    patch_embed  x=[B, 3, 8, 8]  out=[B, 16, 64]
1      n1    relu         x=[B, 16, 64]   out=[B, 16, 64]
```

Parameters: 832
