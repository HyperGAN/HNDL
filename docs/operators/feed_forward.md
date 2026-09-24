# `feed_forward`

Transformer feed-forward block: widen, activate, project back.

**Category:** sequence · **Identity:** `feed_forward@1`

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
| `hidden` (positional) | int | required | >= 1; <= 2147483647 | Width of the inner layer; typically two to four times the feature width D. |
| `activation` | str | `"gelu"` | one of `gelu`, `gelu_tanh`, `relu`, `silu`, `quick_gelu` | Nonlinearity applied to the inner activations. |
| `dropout` | float | `0.0` | >= 0; < 1 | Dropout probability applied after the activation; 0 disables it. |
| `bias` | bool | `True` | — | Add a learned bias to both projections. |
| `equalized` | bool | `False` | — | Use runtime fan-in scaling and N(0,1) raw weights for both linear projections. |

## Description

The position-wise feed-forward network of a transformer block: a widening
projection, a nonlinearity, optional dropout, and a projection back to the
input width.

```text
h   = activation(x @ up.weight.T + up.bias)      # [B, ..., hidden]
out = dropout(h) @ down.weight.T + down.bias     # [B, ..., D]
```

The last axis carries the features `D`; every leading axis is a batch or
position axis, so `[B, D]` and `[B, T, D]` are both accepted and the map is
applied independently at each position. Image tensors `[B, C, H, W]` are
rejected with `E_CONSTRAINT`; flatten or reshape them first.

Submodules are `up` (`Linear(D, hidden)`), `dropout` and `down`
(`Linear(hidden, D)`), so the parameters are `up.weight`, `up.bias`,
`down.weight` and `down.bias`. With biases the block holds
`2 * D * hidden + hidden + D` parameters, and `2 * D * hidden` without.

``equalized=True`` initializes both raw projection weights N(0,1), zeros
their biases, and scales each weight by the inverse square root of its own
fan-in (D for up, hidden for down) on every forward. Gain and learning-rate
multiplier are one. Activations and dropout are unchanged. ``init=`` and
checkpoints contain raw weights; parameter names and shapes stay the same.

`activation` selects one of:

| Name | Formula |
| --- | --- |
| `gelu` | `h * Phi(h)` with the exact Gaussian CDF |
| `gelu_tanh` | the `tanh` approximation of GELU |
| `relu` | `max(h, 0)` |
| `silu` | `h * sigmoid(h)` |
| `quick_gelu` | `h * sigmoid(1.702 * h)` |

Dropout is active in train mode only; in eval mode, and whenever `dropout`
is 0, the block is deterministic. Everything is computed in the incoming
dtype, with no upcasting.

## Examples

### Example 1

The block preserves the feature width, so it drops into any position.

```python
feed_forward(64)
```

Input `['B', 32]` → output `['B', 32]`.

```text
Network: [B, 32] -> [B, 32]  dtype=float32
index  name  operation     input shapes  output shapes
0      n0    feed_forward  x=[B, 32]     out=[B, 32]
```

Parameters: 4,192

### Example 2

Equalized projections preserve the selected activation and dropout.

```python
feed_forward(64, equalized=True)
```

Input `['B', 32]` → output `['B', 32]`.

```text
Network: [B, 32] -> [B, 32]  dtype=float32
index  name  operation     input shapes  output shapes
0      n0    feed_forward  x=[B, 32]     out=[B, 32]
```

Parameters: 4,192

### Example 3

The inner width is explicit; the surrounding widths are inferred.

```python
linear(32)
feed_forward(96, activation="silu")
linear()
```

Input `['B', 16]` → output `['B', 10]`.

```text
Network: [B, 16] -> [B, 10]  dtype=float32
index  name  operation     input shapes  output shapes
0      n0    linear        x=[B, 16]     out=[B, 32]
1      n1    feed_forward  x=[B, 32]     out=[B, 32]
2      n2    linear        x=[B, 32]     out=[B, 10]
```

Parameters: 7,146

### Example 4

On a [B, T, D] sequence the block acts on the last axis, independently per position.

```python
feed_forward(64, activation="gelu_tanh")
```

Input `['B', 8, 24]` → output `['B', 8, 24]`.

```text
Network: [B, 8, 24] -> [B, 8, 24]  dtype=float32
index  name  operation     input shapes  output shapes
0      n0    feed_forward  x=[B, 8, 24]  out=[B, 8, 24]
```

Parameters: 3,160

### Example 5

Without biases the block holds exactly 2 * D * hidden parameters.

```python
feed_forward(32, bias=False)
```

Input `['B', 4, 16]` → output `['B', 4, 16]`.

```text
Network: [B, 4, 16] -> [B, 4, 16]  dtype=float32
index  name  operation     input shapes  output shapes
0      n0    feed_forward  x=[B, 4, 16]  out=[B, 4, 16]
```

Parameters: 1,024
