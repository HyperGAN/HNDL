# `moe`

Sparse mixture of experts: route every token to its top-k feed-forward experts.

**Category:** memory · **Identity:** `moe@1`

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
| `activation` | str | `"gelu"` | one of `gelu`, `gelu_tanh`, `relu`, `silu` | Nonlinearity inside each expert: "gelu", "gelu_tanh", "relu" or "silu". |

## Description

A sparsely gated mixture of experts over the last axis of ``[B, D]`` or
``[B, T, D]`` inputs. Every row of the flattened tensor (one example, or
one position of one example) is a *token* and is routed independently.

```text
logits  = router(x)                        # [N, experts], bias-free linear
v, idx  = topk(logits, top_k)              # the top_k experts per token
w       = softmax(v)                       # renormalized over the selection
out     = sum_k w[:, k] * expert[idx[:, k]](x)
```

Each expert is ``down(activation(up(x)))`` with ``up`` of shape
``[hidden, D]`` and ``down`` of shape ``[D, hidden]``, both with a bias.
The experts live in an ``nn.ModuleList`` named ``experts``, so their
parameters are ``experts.<i>.up.weight``, ``experts.<i>.up.bias``,
``experts.<i>.down.weight`` and ``experts.<i>.down.bias``; the router is
``router.weight``. Because the gate weights are a softmax over only the
selected logits, they sum to one per token, and ``top_k == experts`` makes
the layer a dense softmax-weighted mixture.

Dispatch is a loop over the experts with a boolean row mask, chosen for
clarity over throughput: cost grows with the number of experts even when
each one sees few tokens. The router, the softmax and the experts all
compute in the plan's compute dtype; nothing is upcast. Behavior is
identical in train and eval mode, and routing is deterministic given the
parameters (no noise, no capacity limit, so no token is ever dropped).

Limitation: this operator emits no auxiliary load-balancing loss, so
nothing pushes the router toward using its experts evenly. A host that
wants one can recompute it from the router logits, for example as
``experts * sum_e fraction_of_tokens_to_e * mean_softmax_prob_e``, by
reading ``router`` out of the built model.

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
index  name  operation  input shapes  output shapes
0      n0    linear     x=[B, 8]      out=[B, 16]
1      n1    moe        x=[B, 16]     out=[B, 16]
2      n2    linear     x=[B, 16]     out=[B, 4]
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
index  name  operation  input shapes  output shapes
0      n0    moe        x=[B, 4, 8]   out=[B, 4, 8]
1      n1    linear     x=[B, 4, 8]   out=[B, 4, 3]
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
index  name  operation  input shapes  output shapes
0      n0    moe        x=[B, 6, 12]  out=[B, 6, 12]
```

Parameters: 1,872
