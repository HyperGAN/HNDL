# `spatial_attention`

SAGAN self-attention over every position of a [B, C, H, W] feature map.

**Category:** spatial · **Identity:** `spatial_attention@1`

## Shape

```text
x[B, C, H, W] -> out[B, C, H, W]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H, W]` | compute |
| `out` | output | `out[B, C, H, W]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `reduction` (positional) | int | `8` | >= 1 | How much narrower the query and key projections are than the input; must evenly divide the channel width C. The SAGAN default of 8 gives C/8 query channels. |

## Description

The Self-Attention GAN block on a ``[B, C, H, W]`` feature map: every one
of the ``N = H*W`` positions attends to every other, so the layer sees the
whole map where a 3x3 convolution sees a neighbourhood.

```text
f, g    = f_proj(x), g_proj(x)              # 1x1 convs, C -> C/reduction
h       = h_proj(x)                         # 1x1 conv,  C -> C
f, g, h -> [B, C', N], [B, C', N], [B, C, N]     # N = H*W
energy[b, i, j] = sum_c f[b, c, i] * g[b, c, j]  # [B, N, N]
beta    = softmax(energy, dim=-1)           # over the key positions j
o[b, c, i] = sum_j h[b, c, j] * beta[b, i, j]    # [B, C, N] -> [B, C, H, W]
out     = gamma * o + x
```

``reduction`` must divide ``C``; the resolver reports ``E_CONSTRAINT`` when
it does not. Submodules are ``f_proj`` (query), ``g_proj`` (key) and
``h_proj`` (value), each an ``nn.Conv2d(C, ·, kernel_size=1)`` with a bias,
so their parameters are ``f_proj.weight``, ``f_proj.bias`` and so on.
``f_proj`` and ``g_proj`` project to ``C // reduction`` channels while
``h_proj`` keeps all ``C``. Nothing else is learned: the softmax, the two
contractions and the reshapes carry no state and there are no buffers.

The remaining parameter is ``gamma``, a single learned scalar of shape
``[1]`` that gates the attention branch, exactly as `learned_scale` does.
It starts at ``0.0``, so the block begins as the identity — ``out == x`` on
the first step — and the network learns how much attention to admit. A
gamma of zero also zeroes the gradient reaching the three projections;
they start learning through ``gamma`` itself, whose gradient is the inner
product of the attention branch with the incoming gradient. Construction
overrides target it by name: ``spatial_attention(init={"gamma": 1.0})``
opens the gate and ``spatial_attention(trainable={"gamma": False})``
freezes it.

Shape, rank and dtype are preserved and train and eval behave identically.
Cost grows with ``N**2 = (H*W)**2``: the attention map is ``[B, N, N]``,
which is 256x256 per example on a 16x16 map and 4096x4096 on a 64x64 one,
so the block is normally placed at a middle resolution. ``H`` and ``W`` are
not inferable backward — every spatial extent is accepted — so they come
from the network input or the preceding operation, while ``C`` flows in
both directions.

Zhang, Goodfellow, Metaxas & Odena, "Self-Attention Generative Adversarial
Networks" (ICML 2019, arXiv 2018), section 3: f and g project to
``C/8`` channels, h keeps ``C``, the attention map is
``softmax(f(x)^T g(x))`` over the key positions, and the output is
``y = gamma * o + x`` with gamma initialized to 0.

## Examples

### Example 1

The SAGAN block at its paper settings: 64 channels, C/8 = 8 query channels, 64 positions attending to each other.

```python
spatial_attention()
```

Input `['B', 64, 8, 8]` → output `['B', 64, 8, 8]`.

```text
Network: [B, 64, 8, 8] -> [B, 64, 8, 8]  dtype=float32
index  name  operation          input shapes     output shapes
0      n0    spatial_attention  x=[B, 64, 8, 8]  out=[B, 64, 8, 8]
```

Parameters: 5,201

### Example 2

A small feature map with a gentler reduction: 8 query channels out of 16.

```python
spatial_attention(2)
```

Input `['B', 16, 4, 4]` → output `['B', 16, 4, 4]`.

```text
Network: [B, 16, 4, 4] -> [B, 16, 4, 4]  dtype=float32
index  name  operation          input shapes     output shapes
0      n0    spatial_attention  x=[B, 16, 4, 4]  out=[B, 16, 4, 4]
```

Parameters: 545

### Example 3

A convolutional trunk with one attention block; the 32 channels flow into the block and out of it unchanged.

```python
conv(32, kernel_size=3, padding=1)
spatial_attention(4)
global_avg_pool()
linear()
```

Input `['B', 3, 8, 8]` → output `['B', 10]`.

```text
Network: [B, 3, 8, 8] -> [B, 10]  dtype=float32
index  name  operation          input shapes     output shapes
0      n0    conv               x=[B, 3, 8, 8]   out=[B, 32, 8, 8]
1      n1    spatial_attention  x=[B, 32, 8, 8]  out=[B, 32, 8, 8]
2      n2    global_avg_pool    x=[B, 32, 8, 8]  out=[B, 32]
3      n3    linear             x=[B, 32]        out=[B, 10]
```

Parameters: 2,811
