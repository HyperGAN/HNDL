# `adaln_zero`

Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.

**Category:** normalization · **Identity:** `adaln_zero@1`

## Shape

```text
x[B, T, D], cond[B, D_cond] -> out[B, T, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `cond` | input | `cond[B, D_cond]` | compute |
| `out` | output | `out[B, T, D]` | compute |
| `gate` | output | `gate` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `eps` | float | `1e-06` | > 0 | Epsilon added to the variance in LayerNorm. |
| `gate` | bool | `True` | — | Whether to output gate tensor. |

## Description

Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.

Maps a conditioning vector ``cond[B, D_cond]`` through a zero-initialized
linear layer to produce per-token shift, scale, and optional dimension-wise gate
parameters:
``shift, scale, gate = Linear(SiLU(cond)).chunk(3, dim=-1)`` (when ``gate=True``)
``shift, scale = Linear(SiLU(cond)).chunk(2, dim=-1)`` (when ``gate=False``)
``normed = LayerNorm(x, affine=False)``
``out = (1 + scale) * normed + shift``
``gate_out = gate.expand_as(x)``.

## Examples

### Example 1

Modulates sequence features with conditioning vector and produces residual gate.

```python
c = mean(1)
cond = linear(c, 32)
normed, gate = adaln_zero(x, cond)
mul(normed, gate)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
name  operation   input shapes                                           output shapes
n0    mean        x=input:x:[B, 16, 64]                                  out=[B, 64]
n1    linear      x=node:n0/out:[B, 64]                                  out=[B, 32]
n2    adaln_zero  x=input:x:[B, 16, 64], cond=node:n1/out:[B, 32]        out=[B, 16, 64], gate=[B, 16, 64]
n3    mul         a=node:n2/out:[B, 16, 64], b=node:n2/gate:[B, 16, 64]  out=[B, 16, 64]
```

Parameters: 8,416
