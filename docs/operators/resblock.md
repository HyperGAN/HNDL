# `resblock`

Residual basic block: two 3x3 convolutions with a normalized shortcut.

**Category:** vision · **Identity:** `resblock@1`

## Shape

```text
x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]
```

Relation: `H_out = floor((H_in + 2 - 3) / stride) + 1 = ceil(H_in / stride), same for W`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C_in, H_in, W_in]` | compute |
| `out` | output | `out[B, C_out, H_out, W_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `out_channels` (positional) | int | inferred | >= 1; <= 2147483647; binds `C_out` | Channels the block produces. Omit to infer from the consumer. |
| `in_channels` | int | inferred | >= 1; <= 2147483647; binds `C_in` | Input channels. Normally inferred from the incoming tensor. |
| `stride` | int | `1` | >= 1 | Stride of the first convolution and of the shortcut; 2 halves height and width. |
| `norm` | str | `"batch_norm"` | one of `batch_norm`, `group_norm`, `none` | Normalization after each convolution: "batch_norm", "group_norm", or "none". |
| `groups` | int | `8` | >= 1 | Channel groups when norm="group_norm"; must divide the channel count. Ignored otherwise. |

## Description

The ResNet *basic block* on ``[B, C, H, W]`` images:

```text
y = norm2(conv2(relu(norm1(conv1(x)))))
out = relu(y + shortcut(x))
```

Both convolutions use a 3x3 kernel with padding 1; only ``conv1`` carries
the stride, so height and width map as
``H_out = floor((H_in + 2 - 3) / stride) + 1``, which equals
``ceil(H_in / stride)`` — stride 1 preserves the spatial extent and
stride 2 halves it (rounding up).

The shortcut is an ``nn.Identity`` when ``stride == 1`` and the channel
count is unchanged. Otherwise it is an ``nn.Sequential`` of a 1x1
convolution with the same stride followed by its own normalization, so
the residual sum is well defined. Submodules are named ``conv1``,
``norm1``, ``conv2``, ``norm2`` and ``shortcut``; parameter targets for
``init`` and ``trainable`` use those paths, for example
``init={"norm2.weight": 0}`` for a zero-initialized residual branch.

``norm="batch_norm"`` keeps running mean and variance buffers, so train
and eval mode differ: training normalizes with the statistics of the
current batch and updates the buffers, while evaluation uses the stored
running statistics. ``norm="group_norm"`` normalizes ``groups`` channel
groups per example and behaves identically in both modes; ``groups`` must
divide the channel count. ``norm="none"`` gives a plain residual
convolution block — no normalization at all — and in that case the
convolutions carry a learned bias, which the normalized variants omit
because the following normalization would cancel it.

Convolutions and normalizations run in the plan's compute dtype; batch
normalization accumulates its batch statistics in that dtype as PyTorch
does, so ``float16`` plans inherit the usual reduced-precision behavior.

## Examples

### Example 1

Two blocks; the strided one halves height and width before the classifier head.

```python
resblock(16)
resblock(32, stride=2)
flatten()
linear()
```

Input `['B', 3, 8, 8]` → output `['B', 10]`.

```text
Network: [B, 3, 8, 8] -> [B, 10]  dtype=float32
index  name  operation  input shapes     output shapes
0      n0    resblock   x=[B, 3, 8, 8]   out=[B, 16, 8, 8]
1      n1    resblock   x=[B, 16, 8, 8]  out=[B, 32, 4, 4]
2      n2    flatten    x=[B, 32, 4, 4]  out=[B, 512]
3      n3    linear     x=[B, 512]       out=[B, 10]
```

Parameters: 22,538

### Example 2

Group normalization with 4 groups over 8 channels; batch size never affects the statistics.

```python
resblock(8, norm="group_norm", groups=4)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    resblock   x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
```

Parameters: 864

### Example 3

A plain residual convolution block; the width 16 is inferred from the output contract.

```python
resblock(norm="none")
```

Input `['B', 8, 16, 16]` → output `['B', 16, 16, 16]`.

```text
Network: [B, 8, 16, 16] -> [B, 16, 16, 16]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    resblock   x=[B, 8, 16, 16]  out=[B, 16, 16, 16]
```

Parameters: 3,632
