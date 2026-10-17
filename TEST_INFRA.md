# HNDL SOTA Operators Testing Infrastructure & Verification Architecture

**Version:** 0.9.0-dev  
**Scope:** SOTA Operators Verification Architecture, E2E Test Suites, 4-Tier Methodology, and Feature Coverage  
**Maintainer:** Worker E2E (Testing Infrastructure & Benchmark Engineer)  

---

## 1. Opaque-Box Testing Architecture

### 1.1 Philosophy & Core Principles

The HNDL SOTA testing infrastructure is built on the foundation of **opaque-box, requirement-driven behavioral verification**. Rather than inspecting private module variables or patching internal state dictionaries, all test suites interact strictly with HNDL through its authoritative public API boundaries:
- Declarative DSL & Specification: `hndl.resolve(source, **kwargs)`, `hndl.loads()`, `hndl.dump()`
- PyTorch Runtime & Execution: `hndl.torch.build(plan, device=...)`, `hndl.torch.network(source, ...)`
- Type Contracts & Registries: `hndl.Registry`, `hndl.Operator`, `hndl.ResolvedPlan`, `hndl.Graph`
- Pretrained & Model Tooling: `hndl.pretrained.load()`, `registry.pretrained_provider`
- Public Operators & Helpers: `hndl.ops`, `hndl.relations`, `hndl.testing`

This opaque-box methodology ensures that tests verify authentic behavioral semantics, mathematical correctness, boundary stability, autograd gradient flows, and execution speed. Internal implementation details—such as chunkwise WY parallel representations vs sequential recurrences, fused ATen/Triton GQA dispatches, zero-break `torch.compile` paths, or low-rank gate factorizations—remain interchangeable beneath stable interface contracts.

### 1.2 Verification Invariants

Every test gate enforces four fundamental invariants:
1. **Behavioral Invariance**: Forward evaluation results, output shapes, error codes, and exception types match specification contracts regardless of device (`cpu`, `cuda`) or precision mode.
2. **Mathematical & Numerical Parity**: All operator kernels produce outputs numerically close to mathematical reference models within strict floating-point tolerances (`rtol=1e-4`, `atol=1e-5` for float32/float64; `1e-3` for half-precision/Triton kernels), and satisfy exact analytical Jacobian checks under `gradcheck`.
3. **Autograd Gradient Flow**: Every trainable parameter in the built `GraphModule` receives valid, non-null finite gradients during backward passes, ensuring absence of detached computation paths or dead parameters.
4. **Compilation Compatibility**: Operators and network blocks run cleanly under `torch.compile(..., fullgraph=True)` with zero unexpected graph breaks.

---

## 2. The 4-Tier Testing Methodology

The verification matrix for modern SOTA operators is structured into four distinct, progressive tiers:

```
+-----------------------------------------------------------------------------------+
| Tier 4: Real-World Application Workloads                                          |
| (Hybrid KDA+MLA decoders, Qwen3 GQA trunks, DiT blocks, Audio synthesis stems)    |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Interactions (Pairwise Combinations)                        |
| (GQA+QK-Norm, MLA+NoPE, KDA+SwiGLU MoE, Causal Conv1d+Deconv1d, AdaLN+Sinusoidal) |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases (Limits, Extremes, Defect Triggers)               |
| (Odd lengths, non-multiple chunks, zero scale, max context, extreme dimensions)   |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (Happy Paths in Isolation)                               |
| (>=5 tests per feature area across all new and extended SOTA operators)           |
+-----------------------------------------------------------------------------------+
```

### 2.1 Tier 1: Feature Coverage (Happy Paths in Isolation)
- **Scope**: Verifies primary functional requirements for every new and extended SOTA operator in isolation.
- **Requirement**: At least 5 independent, self-contained test cases per feature area covering:
  - F1: `attention` & `cross_attention` SOTA Extensions (`kv_heads` GQA, `head_dim`, `qk_norm`, `rope_theta`, `window_size`, `logit_softcap`, `cross_attention` biases).
  - F2: `transformer_block` SOTA Extensions (`ffn_kind="swiglu"`, `hidden` width override, `rope`, `rope_theta`, `kv_heads`, `qk_norm`, `bias`).
  - F3: `moe` SOTA Extensions (3-projection `expert_kind="swiglu"`, `bias=False`, `scoring="sigmoid"`, compile compatibility).
  - F4: Convolution, Normalization, & MatMul Extensions (`conv1d` `causal=True` and `channels_last=True`, `rms_norm` `zero_centered=True`, `layer_norm` `bias=False`, `adaptive_norm` rank-3 sequence support, `matmul` multi-rank batching).
  - F5: `gated_delta_attention` (KDA `decay_kind="channel"`, GDN `decay_kind="head"`, causal depthwise conv1d k=4, per-head L2 norm, chunkwise WY forward).
  - F6: `latent_attention` (Multi-Head Latent Attention MLA, `kv_lora_rank`, `q_lora_rank`, decoupled RoPE, NoPE mode `qk_rope_head_dim=0`).
  - F7: Normalization & Positional Operators (`l2_norm` parameter-free, `adaln_zero` rank-3 conditioning returning out & gate, `rope` standalone rotary embedding).
  - F8: Domain-Completing Operators (`deconv1d` with `policy="up2"`, `grn` Global Response Normalization, `sinusoidal_embed` deterministic log-spaced embeddings, `cast` explicit dtype conversion).
  - F9: `pretrained` Provider Contract Hardening (`shape_fn` callback avoiding meta device execution, `contract="ids"` token-ID inputs).
  - F10: Authored SOTA Network Recipes & Architecture Integration (`llama_tiny`, `deepseek_moe_tiny`, `whisper_seq2seq_tiny`, `convnext_tiny`, `espcn_super_res`, `kimi_linear_tiny`, `qwen3_gqa_qknorm`, `dit_tiny`).

### 2.2 Tier 2: Boundary & Corner Cases (Limits, Extremes, Defect Triggers)
- **Scope**: Exercises operator behavior under extreme dimensions, zero values, non-standard divisibility, and error invalidation paths.
- **Requirement**: At least 5 boundary test cases per feature area covering:
  - Empty or single-token sequence contexts ($T=1$, batch=1).
  - Odd sequence lengths with RoPE and sliding windows.
  - Chunk size mismatch in recurrent linear attention (e.g. sequence length not divisible by `chunk_size` or sequence length smaller than `chunk_size`).
  - Zero-scale initializations (`learned_scale` init=0, `rms_norm` zero_centered weight=0, `adaln_zero` zero linear projection).
  - Divisibility constraint violations (`heads % kv_heads != 0` raising `E_CONSTRAINT`, invalid RoPE configurations raising `E_ARGUMENT`, padding on causal conv raising `E_ARGUMENT`).
  - Numeric stability under near-zero norms ($\epsilon$ protection in `l2_norm` and `rms_norm`).
  - Channel dimension inference backward propagation in sub-pixel convolutions without explicit `out_channels`.

### 2.3 Tier 3: Cross-Feature Interactions (Pairwise Combinations)
- **Scope**: Verifies that new and extended operators interact seamlessly without unintended coupling, shape mismatch, or graph resolution conflicts.
- **Requirement**: At least 15 pairwise interaction tests covering:
  - GQA (`kv_heads`) + per-head QK-Norm (`qk_norm="rms_norm"`).
  - Multi-Head Latent Attention (`latent_attention`) + NoPE mode (`qk_rope_head_dim=0`).
  - Kimi Delta Attention (`gated_delta_attention`) + SwiGLU MoE (`moe(expert_kind="swiglu")`).
  - Causal Conv1d (`causal=True, channels_last=True`) + Deconv1d (`policy="up2"`).
  - AdaLN-Zero (`adaln_zero`) + Sinusoidal Embed (`sinusoidal_embed`).
  - GQA + High-Theta RoPE + Logit Soft-Capping.
  - Standalone RoPE + Parameter-free L2-Norm.
  - Global Response Normalization (GRN) + Zero-Centered RMSNorm.
  - Rank-3 AdaptiveNorm + Multi-Rank Batched MatMul.
  - Cast Operator + Embedding + Attention.
  - Causal Conv1d + KDA Depthwise Conv.
  - TransformerBlock SwiGLU + QK-Norm + High-Theta RoPE.
  - Latent Attention + AdaLN-Zero Diffusion Conditioning.
  - Deconv1d Audio Synthesis + CrossAttention Conditioning.
  - SwiGLU MoE with Sigmoid Affinity Routing + LayerNorm(bias=False).

### 2.4 Tier 4: Real-World Application Workloads
- **Scope**: Validates end-to-end multi-layer architectures matching frontier production models with full forward and backward autograd gradient checks.
- **Requirement**: At least 6 real-world workloads covering:
  - Workload 1: Hybrid Linear-Attention Causal LLM Block (3:1 KDA + NoPE MLA + SwiGLU with pre-RMSNorm).
  - Workload 2: Next-Gen GQA Decoder Block (Qwen3 style: 4:2 GQA, per-head QK-RMSNorm, high-theta RoPE 1,000,000, SwiGLU FFN, zero-centered RMSNorm).
  - Workload 3: Diffusion Transformer (DiT) Block (AdaLN-Zero modulation, Sinusoidal timestep embedding, QK-Norm self-attention, and gated residual update).
  - Workload 4: Audio Causal Convolution & Synthesis Trunk (Causal Conv1d acoustic feature extractor, SwiGLU sequence modulation, and Deconv1d waveform upsampler).
  - Workload 5: ConvNeXt-V2 Modern Vision Stage (7x7 depthwise conv, Global Response Normalization GRN, inverted 1x1 pointwise conv, and LearnedScale).
  - Workload 6: DeepSeek-V3 Style Sparse MoE Transformer Block (Multi-Head Latent Attention with compressed KV cache, top-2 routed SwiGLU MoE with Sigmoid affinity routing and bias=False).

---

## 3. Feature Inventory Coverage Matrix

The table below maps all 47 features from `PROJECT.md § Feature Inventory` directly to their testing tiers, primary test classes, and verification strategies:

| # | Feature Name | Tier | Test Class / Scope | Verification Strategy & Invariants |
|---|--------------|------|--------------------|-----------------------------------|
| 1 | `attention` GQA/MQA | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `kv_heads` with fused `enable_gqa=True`, enforces `heads % kv_heads == 0`. |
| 2 | `attention` `head_dim` override | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests decoupled inner width `heads * head_dim != D`, validates output projection to `D`. |
| 3 | `attention` per-head QK normalization | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `qk_norm` supporting `"none"`, `"rms_norm"`, `"l2_norm"` per-head before RoPE. |
| 4 | `attention` configurable `rope_theta` | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests configurable base frequency (e.g. 500,000, 1,000,000), rejects theta != 10000 when rope=False. |
| 5 | `attention` sliding-window attention | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `window_size` sliding band mask in combination with causal and non-causal attention. |
| 6 | `attention` `logit_softcap` | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests $C \cdot \tanh(\text{scores} / C)$ bounded score scaling. |
| 7 | `cross_attention` SOTA extensions | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `qkv_bias`, `out_bias`, `kv_heads`, `head_dim`, `qk_norm`, and `logit_softcap`. |
| 8 | `transformer_block` FFN kind & hidden override | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `ffn_kind="swiglu"` and explicit `hidden` override (e.g. non-integer ratio 8/3 D). |
| 9 | `transformer_block` RoPE & Attention options | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests pass-through of `rope`, `rope_theta`, `kv_heads`, `qk_norm`, and `bias`. |
| 10 | `moe` SwiGLU Gated Experts | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests 3-projection gated experts (`expert_kind="swiglu"`) alongside `"mlp"`. |
| 11 | `moe` Bias & Sigmoid Affinity | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `bias=False` and `scoring="sigmoid"` affinity routing over top-$k$ experts. |
| 12 | `moe` Compile Compatibility & Reference Oracle | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | Zero graph breaks under `torch.compile(fullgraph=True)` and dense reference parity. |
| 13 | `conv1d` Causal & Channels-Last | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `causal=True` left-padding, bijective $L_{out} == L_{in}$ in solver, and `channels_last=True`. |
| 14 | `rms_norm` Zero-Centered Scaling | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `zero_centered=True` Gemma $(1 + w)$ scaling with weight initialized to 0.0. |
| 15 | `layer_norm` Bias Option | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `bias=False` support using `elementwise_affine=True`. |
| 16 | `adaptive_norm` Rank-3 Sequence Support | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests rank-3 `[B, T, C]` sequence modulation alongside rank-4 `[B, C, H, W]`. |
| 17 | `matmul` Multi-Rank Batching | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests generalization to rank 2, 3, and 4 batched matrix multiplication (`[B, ..., N, K] @ [B, ..., K, M]`). |
| 18 | `gated_delta_attention` Operator Definition | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `gated_delta_attention@1` in category `"sequence"` with KDA and Gated DeltaNet decay kinds. |
| 19 | `gated_delta_attention` Input Projections | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests short causal depthwise conv1d ($k=4$), SiLU, per-head L2 Q/K norm, and beta writing gate. |
| 20 | `gated_delta_attention` Output Projections | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | Tests head-wise RMSNorm, low-rank sigmoid output gate, and output projection $W_o$. |
| 21 | `gated_delta_attention` Chunkwise WY Forward | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Pure PyTorch chunkwise parallel Tensor Core evaluation via `torch.linalg.solve_triangular`. |
| 22 | `gated_delta_attention` Sequential Recurrence Reference | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | Step-by-step state recurrence oracle parity across sequence lengths and head counts. |
| 23 | `latent_attention` Operator Definition | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `latent_attention@1` implementing Multi-Head Latent Attention (MLA). |
| 24 | `latent_attention` LoRA Compression & NoPE Mode | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests KV LoRA rank compression, optional Q LoRA rank, decoupled RoPE, and `qk_rope_head_dim=0` NoPE. |
| 25 | `l2_norm` Operator | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `l2_norm@1` parameter-free L2 normalization with `dim` and `eps`. |
| 26 | `adaln_zero` Operator | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `adaln_zero@1` rank-3 adaptive LayerNorm-Zero returning modulated tokens and gate. |
| 27 | `rope` Standalone Operator | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `rope@1` standalone rotary position embedding with `heads` and `theta`. |
| 28 | `deconv1d` Operator | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `conv_transpose1d@1` in category `"convolution"` with `policy="up2"`. |
| 29 | `grn` Operator | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `grn@1` Global Response Normalization for ConvNeXt-V2. |
| 30 | `sinusoidal_embed` Operator | Tier 1, 3 | `TestTier1FeatureCoverage`, `TestTier3CrossFeatureInteractions` | Tests `sinusoidal_embed@1` deterministic log-spaced sinusoidal timestep embeddings. |
| 31 | `cast` Operator | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `cast@1` explicit dtype conversion across edges with resolver propagation. |
| 32 | `pretrained` Explicit Shape Callback | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests optional `shape_fn` on `registry.pretrained_provider` avoiding meta-device tracing. |
| 33 | `pretrained` Token-ID Input Support | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Tests `contract="ids"` (`int64`, rank 2 `[B, T]`) in `LocalProvider` and `pretrained` operator. |
| 34 | `pretrained` Hardening Test Suite | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | Unit and integration testing for Triton/CUDA kernel wrapping and token-ID inputs. |
| 35 | `llama_tiny` Network Recipe | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | 27 nodes, 918,656 parameters, pre-RMSNorm, RoPE, SwiGLU causal LLM. |
| 36 | `deepseek_moe_tiny` Network Recipe | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | 19 nodes, 922,240 parameters, hybrid shared + routed MoE decoder. |
| 37 | `whisper_seq2seq_tiny` Network Recipe | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | 20 nodes, 159,296 parameters, multi-input (spectrogram float32 + tokens int64). |
| 38 | `convnext_tiny` Network Recipe | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | 19 nodes, 76,876 parameters, 7x7 depthwise conv, inverted bottleneck, learned scale. |
| 39 | `espcn_super_res` Network Recipe | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | 6 nodes, 26,796 parameters, sub-pixel conv with backward channel inference. |
| 40 | `kimi_linear_tiny` Network Recipe | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | 3:1 hybrid KDA + NoPE MLA, 956,384 parameters. |
| 41 | `qwen3_gqa_qknorm` Network Recipe | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | GQA (4:2), QK-RMSNorm, high-theta RoPE, SwiGLU, 853,376 parameters. |
| 42 | `dit_tiny` Network Recipe | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | AdaLN-Zero diffusion transformer, 695,472 parameters. |
| 43 | Network Recipe Test Scaling | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCases` | Scaling `test_network_examples.py` from 40 to 64 tests across all recipes. |
| 44 | Global Operator Count Synchronization | Tier 1 | `TestTier1FeatureCoverage` | Catalog synchronization from 69 to 78 built-in operators across 12 categories. |
| 45 | Documentation Generator Update | Tier 1 | `TestTier1FeatureCoverage` | `hndl.docs` generating 79 operator pages and updated `docs/networks.md`. |
| 46 | Repository Hygiene & Changelog | Tier 1 | `TestTier1FeatureCoverage` | Verification of `CHANGELOG.md` unreleased entries and clean git hygiene. |
| 47 | E2E Testing Suite (Tiers 1–4) | All Tiers | `tests/e2e/test_sota_operators_e2e.py` | Complete requirement-driven opaque-box test suite publishing `TEST_READY.md`. |

---

## 4. Verification Commands & Runner Instructions

The test suite is executable via standard `uv` entrypoints:

```bash
# 1. Lint and style check for E2E suite:
uv run ruff check tests/e2e

# 2. Run the full SOTA E2E test suite:
uv run pytest tests/e2e/test_sota_operators_e2e.py -v

# 3. Run individual testing tiers:
uv run pytest tests/e2e/test_sota_operators_e2e.py -k "TestTier1" -v
uv run pytest tests/e2e/test_sota_operators_e2e.py -k "TestTier2" -v
uv run pytest tests/e2e/test_sota_operators_e2e.py -k "TestTier3" -v
uv run pytest tests/e2e/test_sota_operators_e2e.py -k "TestTier4" -v

# 4. Run the entire project test suite:
uv run pytest -m "not benchmark"

# 5. Run documentation generator check:
uv run python -m hndl.docs --check --out docs/operators --networks examples/networks --networks-out docs/networks.md
```
