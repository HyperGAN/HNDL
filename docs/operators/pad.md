# `pad`

Enlarge trailing axes with constant, reflected, or replicated borders.

**Category:** shape · **Identity:** `pad@1`

## Shape

```text
x -> out
```

Relation: `out[axis] == x[axis] + left + right for each padded axis; all other axes equal`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | compute |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `padding` (positional) | ints | `()` | >= 0; <= 2147483647 | (left, right) pairs starting at the last axis, as torch.nn.functional.pad orders them. Non-negative; two values per padded axis. |
| `value` | float | `0.0` | — | Fill value for mode='constant'. |
| `mode` | str | `"constant"` | one of `constant`, `reflect`, `replicate` | Border rule: 'constant' fills with value, 'reflect' mirrors without repeating the edge, 'replicate' repeats the edge. |

Positional values fill `padding`.

## Description

``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

``padding`` lists ``(left, right)`` pairs **starting from the last axis**,
exactly as ``F.pad`` orders them:

- rank 2 ``[B, F]``: ``pad(left, right)`` widens the feature axis ``F``.
- rank 3 ``[B, C, L]`` (equivalently ``[B, T, D]``): ``pad(left, right)``
  widens the last axis ``L``.
- rank 4 ``[B, C, H, W]``: ``pad(left, right, top, bottom)`` widens ``W``
  first and then ``H``.

The batch axis is never padded, so ``padding`` holds at most
``rank - 1`` pairs; anything longer, or an odd number of values, is
``E_ARGUMENT``. Each padded axis grows by ``left + right`` and every other
axis is unchanged, and the relation runs in both directions: a known input
extent fixes the output, and a known output extent fixes the input.

``mode='constant'`` fills the new positions with ``value`` and works for
every supported rank. ``mode='reflect'`` and ``mode='replicate'`` follow
the PyTorch restriction to spatial padding: a rank-3 tensor padded on its
last axis, or a rank-4 tensor padded on ``H`` and ``W``; other
rank/padding combinations are rejected with ``E_ARGUMENT``. ``reflect``
additionally requires each pad width to be smaller than the extent it
mirrors (``E_CONSTRAINT``), because the border itself is not repeated.
``value`` must stay at its default for the non-constant modes, which
PyTorch does not accept.

There are no parameters, behavior is identical in train and eval mode, the
output keeps the input dtype, and the gradient of the padded positions is
discarded (``constant``) or accumulated back onto the mirrored or repeated
source positions (``reflect``, ``replicate``).

## Examples

### Example 1

One column on each side of W and one row on each side of H, so the 3x3 convolution preserves the image size.

```python
pad(1, 1, 1, 1)
conv(4, kernel_size=3)
```

Input `['B', 3, 8, 8]` → output `['B', 4, 8, 8]`.

```text
Network: [B, 3, 8, 8] -> [B, 4, 8, 8]  dtype=float32
index  name  operation  input shapes      output shapes
0      n0    pad        x=[B, 3, 8, 8]    out=[B, 3, 10, 10]
1      n1    conv       x=[B, 3, 10, 10]  out=[B, 4, 8, 8]
```

Parameters: 112

### Example 2

Mirrors two positions onto each end of L in a [B, C, L] signal.

```python
pad(2, 2, mode="reflect")
```

Input `['B', 3, 8]` → output `['B', 3, 12]`.

```text
Network: [B, 3, 8] -> [B, 3, 12]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    pad        x=[B, 3, 8]   out=[B, 3, 12]
```

Parameters: 0

### Example 3

Backward inference: the projection width 8 follows from the padded output.

```python
linear()
pad(1, 1)
```

Input `['B', 16]` → output `['B', 10]`.

```text
Network: [B, 16] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 16]     out=[B, 8]
1      n1    pad        x=[B, 8]      out=[B, 10]
```

Parameters: 136

### Example 4

A one-position border of ones on the right and bottom edges.

```python
pad(0, 1, 0, 1, value=1.0, mode="constant")
```

Input `['B', 2, 4, 4]` → output `['B', 2, 5, 5]`.

```text
Network: [B, 2, 4, 4] -> [B, 2, 5, 5]  dtype=float32
index  name  operation  input shapes    output shapes
0      n0    pad        x=[B, 2, 4, 4]  out=[B, 2, 5, 5]
```

Parameters: 0
