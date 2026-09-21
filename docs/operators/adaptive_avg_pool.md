# `adaptive_avg_pool`

Average-pool [B, C, H, W] images to a fixed output height and width.

**Category:** spatial · **Identity:** `adaptive_avg_pool2d@1`

## Shape

```text
x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]
```

Relation: `H_out, W_out == output_size; C is preserved; H_in, W_in are unconstrained`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, C, H_in, W_in]` | compute |
| `out` | output | `out[B, C, H_out, W_out]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `output_size` (positional) | pair | inferred | >= 1; <= 2147483647 | Target output height and width; an int applies to both. Omit it to read the target from the output contract. |

## Description

Averages each ``[B, C, H_in, W_in]`` feature map over a grid of
``output_size = (H_out, W_out)`` windows, producing
``[B, C, H_out, W_out]``. Window ``i`` along an axis of extent ``n`` with
``k`` outputs covers

```text
[floor(i * n / k), ceil((i + 1) * n / k))
```

so the windows tile the axis, overlap when ``k`` does not divide ``n``,
and repeat a single element when ``k > n`` (an upsampling by replication,
which torch permits). The operation has no parameters and behaves
identically in train and eval mode.

### Determinism and higher-order gradients

Every path produces the values of `torch.nn.functional.adaptive_avg_pool2d`
within floating-point tolerance, but which path runs decides whether the
backward pass is deterministic on CUDA:

| Case | Implementation | Deterministic on CUDA |
| --- | --- | --- |
| `H_in % H_out == 0` and `W_in % W_out == 0` | reshape to `[B, C, H_out, H_in//H_out, W_out, W_in//W_out]`, then `mean` over the two window axes | yes |
| ragged extents, `H_out * W_out <= 64` | the windows written out with slicing and `stack` | yes |
| ragged extents, `H_out * W_out > 64` | `F.adaptive_avg_pool2d` | **no** |

The first two paths are built from `reshape`, `mean`, slicing and `stack`,
none of which accumulate with atomics, so they do not raise under
`torch.use_deterministic_algorithms(True)`, they repeat bit-identical
gradients run to run, and they differentiate to arbitrary order. That
makes the common critic pooling — a power-of-two map pooled to 4x4 —
usable with a gradient penalty, which takes a second derivative through
the pool.

The third path falls back to torch's own kernel, whose CUDA backward
accumulates with atomic adds: it raises ``adaptive_avg_pool2d_backward_cuda
does not have a deterministic implementation`` under
`torch.use_deterministic_algorithms(True)`, and its gradients are only
reproducible run to run on the CPU. It is reached only by a ragged pool to
more than 64 windows; pool to a divisor of the input extents, or to a
smaller grid, to stay on a deterministic path.

Shape inference runs forward only for the spatial axes: ``H_out`` and
``W_out`` come from ``output_size``, but ``H_in`` and ``W_in`` are *not*
inferable backward, because every input extent maps to the requested
output. Give the input extents from the network input or the preceding
operation. The channel count ``C`` is preserved and flows both ways, and
``output_size`` itself can be inferred backward from a known output shape.

The average is accumulated in the input dtype; in ``float16`` a very large
window loses precision, so pool in two stages if that matters.

## Examples

### Example 1

Pooling to 1x1 turns the feature map into one value per channel.

```python
conv(16, kernel_size=3, padding=1)
relu()
adaptive_avg_pool(1)
flatten()
linear()
```

Input `['B', 3, 28, 28]` → output `['B', 10]`.

```text
Network: [B, 3, 28, 28] -> [B, 10]  dtype=float32
index  name  operation          input shapes       output shapes
0      n0    conv               x=[B, 3, 28, 28]   out=[B, 16, 28, 28]
1      n1    relu               x=[B, 16, 28, 28]  out=[B, 16, 28, 28]
2      n2    adaptive_avg_pool  x=[B, 16, 28, 28]  out=[B, 16, 1, 1]
3      n3    flatten            x=[B, 16, 1, 1]    out=[B, 16]
4      n4    linear             x=[B, 16]          out=[B, 10]
```

Parameters: 618

### Example 2

A ragged 15x15 map becomes a fixed 4x4 grid, so the classifier head has a fixed width.

```python
adaptive_avg_pool(4)
flatten()
linear()
```

Input `['B', 8, 15, 15]` → output `['B', 5]`.

```text
Network: [B, 8, 15, 15] -> [B, 5]  dtype=float32
index  name  operation          input shapes      output shapes
0      n0    adaptive_avg_pool  x=[B, 8, 15, 15]  out=[B, 8, 4, 4]
1      n1    flatten            x=[B, 8, 4, 4]    out=[B, 128]
2      n2    linear             x=[B, 128]        out=[B, 5]
```

Parameters: 645

### Example 3

The omitted output_size is inferred as (3, 3) from the output contract.

```python
adaptive_avg_pool()
```

Input `['B', 6, 9, 9]` → output `['B', 6, 3, 3]`.

```text
Network: [B, 6, 9, 9] -> [B, 6, 3, 3]  dtype=float32
index  name  operation          input shapes    output shapes
0      n0    adaptive_avg_pool  x=[B, 6, 9, 9]  out=[B, 6, 3, 3]
```

Parameters: 0

### Example 4

A pair pools height and width to different extents.

```python
adaptive_avg_pool((2, 3))
```

Input `['B', 4, 8, 8]` → output `['B', 4, 2, 3]`.

```text
Network: [B, 4, 8, 8] -> [B, 4, 2, 3]  dtype=float32
index  name  operation          input shapes    output shapes
0      n0    adaptive_avg_pool  x=[B, 4, 8, 8]  out=[B, 4, 2, 3]
```

Parameters: 0
