# `adaptive_norm`

Instance-normalize features, then apply a per-example learned scale and bias.

**Category:** normalization · **Identity:** `adaptive_norm@1`

## Shape

```text
x, params -> out
```

Relation: `x[B, C, H, W] or x[B, T, C]; params[B, 2*C]; out matches x`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `params` | input | `params` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `eps` | float | `1e-05` | > 0 | Added to the variance for stability. |

## Description

Population instance normalization followed by a learned affine that the
``params`` tensor supplies per example: ``params = [delta_gamma, beta]``
with one value of each per channel.

```text
normalized = (x - mean(x, HW)) / sqrt(var(x, HW) + eps)      # rank 4
normalized = (x - mean(x, C)) / sqrt(var(x, C) + eps)        # rank 3
out = (1 + delta_gamma) * normalized + beta
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

### Example 3

Sequence modulation: 32 feature channels require 64 style parameters.

```python
tokens = linear(32)
style = pool_tokens(linear(64), 'mean')
adaptive_norm(tokens, style)
```

Input `['B', 16, 16]` → output `['B', 16, 32]`.

```text
Network: [B, 16, 16] -> [B, 16, 32]  dtype=float32
name  operation      input shapes                                           output shapes
n0    linear         x=input:x:[B, 16, 16]                                  out=[B, 16, 32]
n1    linear         x=node:n0/out:[B, 16, 32]                              out=[B, 16, 64]
n2    pool_tokens    x=node:n1/out:[B, 16, 64]                              out=[B, 64]
n3    adaptive_norm  x=node:n0/out:[B, 16, 32], params=node:n2/out:[B, 64]  out=[B, 16, 32]
```

Parameters: 2,656
