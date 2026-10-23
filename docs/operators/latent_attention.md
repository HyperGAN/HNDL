# `latent_attention`

Multi-Head Latent Attention (MLA) with compressed KV cache and decoupled RoPE.

**Category:** sequence · **Identity:** `latent_attention@1`

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
| `heads` (positional) | int | required | >= 1 | Number of query attention heads. |
| `kv_lora_rank` (positional) | int | required | >= 1 | Compressed KV latent dimension d_c (e.g. 512 or D // 4). |
| `q_lora_rank` | int | `0` | >= 0 | Low-rank Q compression rank; 0 disables Q compression. |
| `qk_nope_head_dim` | int | `0` | >= 0 | NoPE query/key head dimension; 0 defaults to D // heads. |
| `qk_rope_head_dim` | int | `0` | >= 0 | Decoupled RoPE head dimension; 0 enables NoPE mode (Kimi Linear), >0 must be even. |
| `v_head_dim` | int | `0` | >= 0 | Value head dimension; 0 defaults to qk_nope_head_dim. |
| `causal` | bool | `True` | — | Apply causal autoregressive attention mask. |
| `rope_theta` | float | `10000.0` | > 1.0 | RoPE base frequency theta. |
| `rope_scaling` | str | `"none"` | one of `none`, `linear`, `ntk`, `yarn`, `llama3` | Frequency scaling algorithm for RoPE context extension. |
| `rope_scale` | float | `1.0` | > 0.0 | Frequency scale factor for context extension. |
| `spatial_shape` | pair | `(0, 0)` | >= 0 | Token grid (H, W) for 2D axial RoPE, row-major with H*W == T. |
| `bias` | bool | `False` | — | Add learned bias to all projections. |
| `softmax_scale` | float | `0.0` | >= 0.0 | Softmax scale factor; 0 defaults to 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim). |

## Description

Multi-Head Latent Attention (MLA) with low-rank KV compression and decoupled RoPE.

Introduced in DeepSeek-V2/V3 and Kimi Linear, MLA down-projects keys and
values into a shared latent vector ``c_kv = RMSNorm(W_DKV x)`` of rank
``kv_lora_rank``, which is up-projected into non-positional content keys
and values. A separate decoupled RoPE key vector ``k_rope`` is shared
across all query heads when ``qk_rope_head_dim > 0``, while setting
``qk_rope_head_dim = 0`` activates position-free NoPE mode.

## Examples

### Example 1

MLA with 4 heads, KV LoRA rank 32, and decoupled RoPE.

```python
latent_attention(4, 32, qk_nope_head_dim=16, qk_rope_head_dim=16, causal=True)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
index  name  operation         input shape  output shape
0      n0    latent_attention  [B, 16, 64]  [B, 16, 64]
```

Parameters: 19,488

### Example 2

NoPE MLA without positional embeddings for hybrid linear-attention models.

```python
latent_attention(4, 32, qk_rope_head_dim=0, causal=True)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
index  name  operation         input shape  output shape
0      n0    latent_attention  [B, 16, 64]  [B, 16, 64]
```

Parameters: 14,368

### Example 3

Full low-rank Q and KV compression MLA.

```python
latent_attention(4, 32, q_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16)
```

Input `['B', 16, 64]` → output `['B', 16, 64]`.

```text
Network: [B, 16, 64] -> [B, 16, 64]  dtype=float32
index  name  operation         input shape  output shape
0      n0    latent_attention  [B, 16, 64]  [B, 16, 64]
```

Parameters: 17,472
