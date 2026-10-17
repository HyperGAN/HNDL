# `moe`

Sparse mixture of experts: route every token to its top-k feed-forward experts.

**Category:** sequence · **Identity:** `moe@1`

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
| `experts` (positional) | int | required | >= 1; <= 2147483647 | Number of feed-forward experts. |
| `hidden` (positional) | int | required | >= 1; <= 2147483647 | Inner width of every expert's feed-forward. |
| `top_k` | int | `2` | >= 1 | Experts each token is routed to; must not exceed experts. |
| `activation` | str | `"gelu"` | one of `gelu`, `gelu_tanh`, `relu`, `silu`, `quick_gelu` | Nonlinearity inside each expert: "gelu", "gelu_tanh", "relu" or "silu". |
| `expert_kind` | str | `"mlp"` | one of `mlp`, `swiglu` | Expert architecture: "mlp" (2-layer) or "swiglu" (3-projection gated). |
| `bias` | bool | `True` | — | Enable learned biases in expert projections. |
| `scoring` | str | `"softmax"` | one of `softmax`, `sigmoid` | Routing gate affinity normalization: "softmax" or "sigmoid". |

## Description

A sparsely gated mixture of experts over the last axis of ``[B, D]`` or
``[B, T, D]`` inputs. Every row of the flattened tensor (one example, or
one position of one example) is a *token* and is routed independently.

```text
logits  = router(x)                        # [N, experts], bias-free linear
v, idx  = topk(logits, top_k)              # the top_k experts per token
w       = softmax(v) or sigmoid(v)/sum     # renormalized over the selection
out     = sum_k w[:, k] * expert[idx[:, k]](x)
```

Each expert is either ``down(activation(up(x)))`` (for ``expert_kind="mlp"``)
or ``down(silu(gate(x)) * up(x))`` (for ``expert_kind="swiglu"``).
The experts live in an ``nn.ModuleList`` named ``experts``; the router is
``router.weight``.

## Examples

### Example 1

Four experts, two of which see each example; the mixture keeps the width at 16.

```python
linear(16)
moe(4, 32)
linear()
```

Input `['B', 8]` → output `['B', 4]`.

```text
Network: [B, 8] -> [B, 4]  dtype=float32
index  name  operation  input shape  output shape
0      n0    linear     [B, 8]       [B, 16]
1      n1    moe        [B, 16]      [B, 16]
2      n2    linear     [B, 16]      [B, 4]
```

Parameters: 4,564

### Example 2

On a [B, T, D] sequence every position is routed independently; top_k=1 is hard routing.

```python
moe(2, 16, top_k=1)
linear()
```

Input `['B', 4, 8]` → output `['B', 4, 3]`.

```text
Network: [B, 4, 8] -> [B, 4, 3]  dtype=float32
index  name  operation  input shape  output shape
0      n0    moe        [B, 4, 8]    [B, 4, 8]
1      n1    linear     [B, 4, 8]    [B, 4, 3]
```

Parameters: 603

### Example 3

The operator preserves its input shape, so it drops into a residual stack unchanged.

```python
moe(3, 24, activation="silu")
```

Input `['B', 6, 12]` → output `['B', 6, 12]`.

```text
Network: [B, 6, 12] -> [B, 6, 12]  dtype=float32
index  name  operation  input shape  output shape
0      n0    moe        [B, 6, 12]   [B, 6, 12]
```

Parameters: 1,872

### Example 4

Four bias-free SwiGLU experts routed with top-2 softmax gating.

```python
moe(4, 32, expert_kind="swiglu", bias=False)
```

Input `['B', 8]` → output `['B', 8]`.

```text
Network: [B, 8] -> [B, 8]  dtype=float32
index  name  operation  input shape  output shape
0      n0    moe        [B, 8]       [B, 8]
```

Parameters: 3,104

### Example 5

Top-2 routing with sigmoid affinity scoring normalized over selected experts.

```python
moe(4, 32, scoring="sigmoid")
```

Input `['B', 8]` → output `['B', 8]`.

```text
Network: [B, 8] -> [B, 8]  dtype=float32
index  name  operation  input shape  output shape
0      n0    moe        [B, 8]       [B, 8]
```

Parameters: 2,240
