# `adaptive_norm`

Instance-normalize features, then apply a per-example learned scale and bias.

**Category:** normalization · **Identity:** `adaptive_norm@1`

## Shape

```text
x[B, C, H, W], params[B, 2*C] -> out[B, C, H, W]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H, W]` | compute |
| `params` | input | `params[B, 2*C]` | compute |
| `out` | output | `out[B, C, H, W]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `eps` | float | `1e-05` | > 0 | Added to the variance for stability. |

## Description

Population instance normalization followed by a learned affine that the
``params`` tensor supplies per example: ``params = [delta_gamma, beta]``
with one value of each per channel.

```text
normalized = (x - mean(x, HW)) / sqrt(var(x, HW) + eps)
out = (1 + delta_gamma)[:, :, None, None] * normalized + beta[:, :, None, None]
```

A zero ``params`` vector yields the normalized features. There are no
running statistics and no parameters; train and eval behave identically.

## Examples

### Example 1

32 channels need 64 style values; the projection resolves to 512.

```python
z1, z2 = split(64)
linear(z1)
features = reshape(32, 4, 4)
adaptive_norm(features, z2)
```

Input `['B', 128]` → output `['B', 32, 4, 4]`.

```text
Network: [B, 128] -> [B, 32, 4, 4]  dtype=float32
name  operation      input shapes                                              output shapes
n0    split          x=input:x:[B, 128]                                        first=[B, 64], rest=[B, 64]
n1    linear         x=node:n0/first:[B, 64]                                   out=[B, 512]
n2    reshape        x=node:n1/out:[B, 512]                                    out=[B, 32, 4, 4]
n3    adaptive_norm  x=node:n2/out:[B, 32, 4, 4], params=node:n0/rest:[B, 64]  out=[B, 32, 4, 4]
```

Parameters: 33,280

### Example 2

A mapping network fans out into a projection and a zero-initialized style affine.

```python
linear(256, name="mapping")
w = relu(name="w")
linear(w, name="project")
features = reshape(64, 4, 4, name="seed")
style = linear(w, name="style", init={"weight": 0, "bias": 0})
adaptive_norm(features, style, name="norm")
```

Input `['B', 128]` → output `['B', 64, 4, 4]`.

```text
Network: [B, 128] -> [B, 64, 4, 4]  dtype=float32
name     operation      input shapes                                                   output shapes
mapping  linear         x=input:x:[B, 128]                                             out=[B, 256]
w        relu           x=node:mapping/out:[B, 256]                                    out=[B, 256]
project  linear         x=node:w/out:[B, 256]                                          out=[B, 1024]
seed     reshape        x=node:project/out:[B, 1024]                                   out=[B, 64, 4, 4]
style    linear         x=node:w/out:[B, 256]                                          out=[B, 128]
norm     adaptive_norm  x=node:seed/out:[B, 64, 4, 4], params=node:style/out:[B, 128]  out=[B, 64, 4, 4]
```

Parameters: 329,088
