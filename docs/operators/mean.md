# `mean`

Average one non-batch axis away.

**Category:** shape · **Identity:** `mean@1`

## Shape

```text
x -> out
```

Relation: `out drops axis dim, or holds 1 there with keepdim; every other axis is unchanged`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `dim` (positional) | int | required | >= 1; <= 3 | Axis to average over. The batch axis 0 cannot be reduced, so dim starts at 1. |
| `keepdim` | bool | `False` | — | Keep the reduced axis with extent 1 instead of dropping it. |

## Description

Arithmetic mean over exactly one non-batch axis:
``out = x.mean(dim=dim, keepdim=keepdim)``.

With ``keepdim=False`` (the default) the output rank is one lower than the
input rank and the axes after ``dim`` shift down by one. With
``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
``[B, T, D]`` (``mean(1)`` pools over positions, ``mean(2)`` over
features), rank 4 is ``[B, C, H, W]`` (``mean(1)`` pools over channels,
``mean(2)`` over height, ``mean(3)`` over width), and a 1-D convolution
layout ``[B, C, L]`` reduces its length with ``mean(2)``.

Because HNDL tensors always keep a batch axis and at least one further
axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
``E_CONSTRAINT``; use ``keepdim=True`` instead.

The layer has no parameters and behaves identically in train and eval
mode. The reduction runs in the incoming dtype — under ``float16`` a long
axis accumulates rounding error, which ``sum`` shares.

Shape inference is bidirectional: a known output fixes every non-reduced
input axis, but the extent of the reduced axis itself carries no
information backward and must come from the input side.

## Examples

### Example 1

Mean-pools a [B, T, D] sequence over its positions into [B, D].

```python
linear(8)
mean(1)
```

Input `['B', 4, 16]` → output `['B', 8]`.

```text
Network: [B, 4, 16] -> [B, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 16]  out=[B, 4, 8]
1      n1    mean       x=[B, 4, 8]   out=[B, 8]
```

Parameters: 136

### Example 2

Two reductions average an image over H and then W: global average pooling.

```python
conv(8, kernel_size=3, padding=1)
mean(2)
mean(2)
```

Input `['B', 3, 8, 8]` → output `['B', 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    mean       x=[B, 8, 8, 8]  out=[B, 8, 8]
2      n2    mean       x=[B, 8, 8]     out=[B, 8]
```

Parameters: 224

### Example 3

With keepdim the reduced axis stays with extent 1, so the rank is preserved.

```python
mean(2, keepdim=True)
```

Input `['B', 3, 8, 8]` → output `['B', 3, 1, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 3, 1, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    mean       x=[B, 3, 8, 8]  out=[B, 3, 1, 8]
```

Parameters: 0

### Example 4

The features surviving the reduction are inferred backward from the output contract.

```python
mean(1)
linear()
```

Input `['B', 6, 32]` → output `['B', 10]`.

```text
Network: [B, 6, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    mean       x=[B, 6, 32]  out=[B, 32]
1      n1    linear     x=[B, 32]     out=[B, 10]
```

Parameters: 330
