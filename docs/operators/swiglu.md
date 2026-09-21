# `swiglu`

SwiGLU feed-forward block: a SiLU-gated projection folded back to the input width.

**Category:** sequence · **Identity:** `swiglu@1`

## Shape

```text
x[B, ..., D] -> out[B, ..., D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, ..., D]` | compute |
| `out` | output | `out[B, ..., D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `hidden` (positional) | int | required | >= 1; <= 2147483647 | Width of the gated inner layer; both inner projections use it. |
| `bias` | bool | `False` | — | Add a learned bias to all three projections. Gated blocks usually omit it. |

## Description

The gated feed-forward block used by PaLM- and LLaMA-style transformers.
Two projections read the same input: one is passed through SiLU and gates
the other elementwise, and a third projection folds the product back to the
input width.

```text
out = down(silu(gate(x)) * up(x))
silu(v) = v * sigmoid(v)
```

The last axis carries the features `D`; every leading axis is a batch or
position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
applied independently at each position. Image tensors `[B, C, H, W]` are
rejected with `E_CONSTRAINT`; flatten or reshape them first.

Submodules are `gate` and `up` (both `Linear(D, hidden)`) and `down`
(`Linear(hidden, D)`). Without biases the block holds `3 * D * hidden`
parameters; with `bias=True` it holds `3 * D * hidden + 2 * hidden + D`.
Because a gated block spends three matrices where a plain `feed_forward`
spends two, `hidden` is commonly set to about two thirds of the width a
plain block would use, for equal parameter count.

There is no dropout and no normalization here, and no running state, so
train and eval mode behave identically. Everything is computed in the
incoming dtype, with no upcasting.

## Examples

### Example 1

The block preserves the feature width, so it drops into any position.

```python
swiglu(64)
```

Input `['B', 32]` → output `['B', 32]`.

```text
Network: [B, 32] -> [B, 32]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    swiglu     x=[B, 32]     out=[B, 32]
```

Parameters: 6,144

### Example 2

The inner width is explicit; the surrounding widths are inferred.

```python
linear(32)
swiglu(96)
linear()
```

Input `['B', 16]` → output `['B', 10]`.

```text
Network: [B, 16] -> [B, 10]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 16]     out=[B, 32]
1      n1    swiglu     x=[B, 32]     out=[B, 32]
2      n2    linear     x=[B, 32]     out=[B, 10]
```

Parameters: 10,090

### Example 3

On a [B, T, D] sequence the block acts on the last axis, independently per position.

```python
swiglu(48, bias=True)
```

Input `['B', 8, 24]` → output `['B', 8, 24]`.

```text
Network: [B, 8, 24] -> [B, 8, 24]  dtype=float32
index  name  operation  input shapes  output shapes
0      n0    swiglu     x=[B, 8, 24]  out=[B, 8, 24]
```

Parameters: 3,576
