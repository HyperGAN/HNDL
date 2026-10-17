# `gated_delta_attention`

Kimi Delta Attention (KDA) and Gated DeltaNet with chunkwise parallel WY evaluation.

**Category:** sequence · **Identity:** `gated_delta_attention@1`

## Shape

```text
x[B, T, D] -> out[B, T, D]
```

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x[B, T, D]` | compute |
| `out` | output | `out[B, T, D]` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `heads` (positional) | int | required | >= 1 | Number of attention heads; must divide width D when head_dim=0. |
| `head_dim` | int | `0` | >= 0 | Dimension per head; 0 defaults to D // heads. |
| `decay_kind` | str | `"channel"` | one of `channel`, `head` | Decay granularity: "channel" for Kimi KDA, "head" for Gated DeltaNet. |
| `gate_rank` | int | `0` | >= 0 | Low-rank bottleneck rank for decay and output gates; 0 defaults to max(head_dim, D // 16). |
| `conv_size` | int | `4` | >= 0; <= 16 | Causal depthwise 1D conv kernel size on Q/K/V; 0 disables convolution. |
| `chunk_size` | int | `64` | >= 1; <= 512 | Chunk size for parallel WY/DPLR Tensor Core evaluation in forward(). |
| `bias` | bool | `False` | — | Add learned bias to input and output projections. |
| `eps` | float | `1e-06` | > 0 | Epsilon for L2 norm and RMSNorm. |

## Description

Kimi Delta Attention (KDA) and Gated DeltaNet associative memory attention.

Evaluates associative memory updates with error-correction delta rule:
residual error ``e_t = v_t - S_{t-1}^T k_t`` is written into memory state ``S_t``.
In ``forward()``, native PyTorch Tensor Core chunkwise parallel WY representation
evaluates intra-chunk dependencies via ``torch.linalg.solve_triangular``, while
inter-chunk states are propagated across chunks.

## Examples

### Example 1

Kimi Delta Attention with channel-wise decay and chunk size 32.

```python
gated_delta_attention(4, decay_kind="channel", chunk_size=32)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
index  name  operation              input shape  output shape
0      n0    gated_delta_attention  [B, 16, 64]  [B, 16, 64]
```

Parameters: 21,888

### Example 2

Gated DeltaNet with scalar head decay and chunk size 32.

```python
gated_delta_attention(4, decay_kind="head", chunk_size=32)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
index  name  operation              input shape  output shape
0      n0    gated_delta_attention  [B, 16, 64]  [B, 16, 64]
```

Parameters: 19,972

### Example 3

Gated delta attention with explicit head_dim and convolution disabled.

```python
gated_delta_attention(2, head_dim=16, conv_size=0, chunk_size=16)
```

Input `['B', 12, 32]` → output `['B', 12, 32]`.

```text
Network: [B, 12, 32] -> [B, 12, 32]  dtype=float32
index  name  operation              input shape  output shape
0      n0    gated_delta_attention  [B, 12, 32]  [B, 12, 32]
```

Parameters: 6,304
