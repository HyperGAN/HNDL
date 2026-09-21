# `global_avg_pool`

Average each channel over height and width, producing one value per channel.

**Category:** spatial · **Identity:** `global_avg_pool@1`

## Shape

```text
x[B, C, H, W] -> out[B, C]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H, W]` | compute |
| `out` | output | `out[B, C]` | compute |

## Arguments

This operator takes no scalar arguments.

## Description

Computes ``out[b, c] = mean(x[b, c, :, :])`` on a ``[B, C, H, W]``
input, returning ``[B, C]``: the spatial axes are reduced away rather
than kept as size-1 axes, so the result feeds `linear` directly.

This is `adaptive_avg_pool(1)` followed by `flatten`, written as one
operation because the pattern is the standard head of a convolutional
classifier. There are no parameters and train and eval behave
identically.

Only rank-4 inputs are accepted, which keeps the axis convention
unambiguous; reduce a ``[B, C, L]`` sequence with a `reshape` to
``[B, C, L, 1]`` first, or use `adaptive_avg_pool`.

Shape inference: ``C`` flows in both directions, while ``H`` and ``W``
are *not* inferable backward — every spatial extent produces the same
output — so they must come from the network input or the preceding
operation. The mean is accumulated in the input dtype.

## Examples

### Example 1

A convolutional trunk followed by a global-pooled classifier head; the head's in_features resolves to the 16 channels.

```python
conv(16, kernel_size=3, padding=1)
relu()
global_avg_pool()
linear()
```

Input `['B', 3, 16, 16]` → output `['B', 10]`.

```text
Network: [B, 3, 16, 16] -> [B, 10]  dtype=float32
index  name  operation        input shapes       output shapes
0      n0    conv             x=[B, 3, 16, 16]   out=[B, 16, 16, 16]
1      n1    relu             x=[B, 16, 16, 16]  out=[B, 16, 16, 16]
2      n2    global_avg_pool  x=[B, 16, 16, 16]  out=[B, 16]
3      n3    linear           x=[B, 16]          out=[B, 10]
```

Parameters: 618

### Example 2

Any spatial extent collapses to one value per channel.

```python
global_avg_pool()
```

Input `['B', 32, 7, 7]` → output `['B', 32]`.

```text
Network: [B, 32, 7, 7] -> [B, 32]  dtype=float32
index  name  operation        input shapes     output shapes
0      n0    global_avg_pool  x=[B, 32, 7, 7]  out=[B, 32]
```

Parameters: 0

### Example 3

Spatial size no longer constrains the head, so the trunk can downsample freely.

```python
conv(8, kernel_size=4, stride=2, padding=1)
group_norm(4)
relu()
global_avg_pool()
linear()
```

Input `['B', 3, 32, 32]` → output `['B', 4]`.

```text
Network: [B, 3, 32, 32] -> [B, 4]  dtype=float32
index  name  operation        input shapes      output shapes
0      n0    conv             x=[B, 3, 32, 32]  out=[B, 8, 16, 16]
1      n1    group_norm       x=[B, 8, 16, 16]  out=[B, 8, 16, 16]
2      n2    relu             x=[B, 8, 16, 16]  out=[B, 8, 16, 16]
3      n3    global_avg_pool  x=[B, 8, 16, 16]  out=[B, 8]
4      n4    linear           x=[B, 8]          out=[B, 4]
```

Parameters: 444
