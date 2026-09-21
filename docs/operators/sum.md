# `sum`

Add up one non-batch axis.

**Category:** shape · **Identity:** `sum@1`

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
| `dim` (positional) | int | required | >= 1; <= 3 | Axis to sum over. The batch axis 0 cannot be reduced, so dim starts at 1. |
| `keepdim` | bool | `False` | — | Keep the reduced axis with extent 1 instead of dropping it. |

## Description

Sum over exactly one non-batch axis:
``out = x.sum(dim=dim, keepdim=keepdim)``.

With ``keepdim=False`` (the default) the output rank is one lower than the
input rank and the axes after ``dim`` shift down by one. With
``keepdim=True`` the rank is preserved and the reduced axis has extent 1.

Axis conventions follow the rest of HNDL: rank 2 is ``[B, F]``, rank 3 is
``[B, T, D]`` (``sum(1)`` adds up positions, ``sum(2)`` adds up features),
rank 4 is ``[B, C, H, W]`` (``sum(1)`` over channels, ``sum(2)`` over
height, ``sum(3)`` over width), and a 1-D convolution layout ``[B, C, L]``
reduces its length with ``sum(2)``.

Because HNDL tensors always keep a batch axis and at least one further
axis, reducing a rank-2 tensor without ``keepdim`` is rejected with
``E_CONSTRAINT``; use ``keepdim=True`` instead.

The layer has no parameters and behaves identically in train and eval
mode. The reduction accumulates in the incoming dtype and is not upcast,
so a long axis in ``float16`` can lose precision or overflow; scale the
inputs, use ``mean``, or run the plan in ``bfloat16``/``float32`` when
that matters.

Shape inference is bidirectional: a known output fixes every non-reduced
input axis, but the extent of the reduced axis itself carries no
information backward and must come from the input side.

## Examples

### Example 1

Adds the positions of a [B, T, D] sequence together into [B, D].

```python
linear(8)
sum(1)
```

Input `['B', 4, 16]` → output `['B', 8]`.

```text
Network: [B, 4, 16] -> [B, 8]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 4, 16]  out=[B, 4, 8]
1      n1    sum        x=[B, 4, 8]   out=[B, 8]
```

Parameters: 136

### Example 2

Collapses the width of an image feature map, leaving [B, C, H].

```python
conv(8, kernel_size=3, padding=1)
sum(3)
```

Input `['B', 3, 8, 8]` → output `['B', 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 8, 8]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    conv       x=[B, 3, 8, 8]  out=[B, 8, 8, 8]
1      n1    sum        x=[B, 8, 8, 8]  out=[B, 8, 8]
```

Parameters: 224

### Example 3

With keepdim the reduced axis stays with extent 1, so the rank is preserved.

```python
sum(1, keepdim=True)
```

Input `['B', 4, 6]` → output `['B', 1, 6]`.

```text
Network: [B, 4, 6] -> [B, 1, 6]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    sum        x=[B, 4, 6]   out=[B, 1, 6]
```

Parameters: 0

### Example 4

The features surviving the reduction are inferred backward from the output contract.

```python
sum(1)
linear()
```

Input `['B', 6, 32]` → output `['B', 10]`.

```text
Network: [B, 6, 32] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    sum        x=[B, 6, 32]  out=[B, 32]
1      n1    linear     x=[B, 32]     out=[B, 10]
```

Parameters: 330
