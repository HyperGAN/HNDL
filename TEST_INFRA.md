# HNDL v0.10.0 Testing Infrastructure & Verification Architecture

**Version:** 0.10.0-dev  
**Scope:** Complete v0.10.0 Verification Architecture, E2E Test Suite (Tiers 1–4 across all 5 Pillars: R1 Bug Remediation, R2 Performance, R3 SOTA Catalog & Solver Extensions, R4 Extensibility & Ecosystem, R5 DX & Lifecycle Primitives)  
**Maintainer:** Worker E2E (Testing Infrastructure & Benchmark Engineer)  

---

## 1. Opaque-Box Testing Architecture

### 1.1 Philosophy & Core Principles

The HNDL v0.10.0 testing infrastructure is founded on **opaque-box, requirement-driven behavioral verification**. Rather than inspecting private module variables or relying on brittle monkey-patches, all test suites interact strictly with HNDL through authoritative public API boundaries:
- **Declarative DSL & Specification:** `hndl.resolve(source, **kwargs)`, `hndl.loads()`, `hndl.dump()`, and `.hndl` text files with scalar constants, `def` macros, and tuple loops.
- **PyTorch Runtime & Execution:** `hndl.torch.build(plan, device=...)`, `hndl.torch.network(source, ...)`, `GraphModule`, `model.param_groups()`, `build(..., checkpoint=...)`, `build(..., init_policy="depth_scaled")`, and `InferenceCache`.
- **Type Contracts & Registries:** `hndl.Registry`, `hndl.Operator`, `hndl.ResolvedPlan`, `hndl.Graph`, and `hndl.Node`.
- **Pretrained & Ecosystem:** `hndl.torch.load_weights(model, path, key_map=...)`, `registry.pretrained_provider`, `plan.to_python_source()`, `model.allow_state_mutation()`, and entry-point plugin discovery.
- **Static Profiler & Tooling:** `plan.profile()`, `plan.to_mermaid()`, `plan.diff()`, and the `hndl` CLI (`inspect`, `mermaid`, `check`).
- **Public Operators & Capture:** `hndl.ops`, `hndl.relations`, `hndl.testing`, `ops.__dir__()`, `ops.__signature__`, and `src/hndl/capture.pyi`.

This methodology ensures that tests verify genuine behavior, mathematical precision, boundary stability, autograd gradient flows, and execution performance. Internal implementation details—such as grouped GEMM dispatches, chunked Tensor Core semi-separable recurrences, topological slot recycling, or fuzzy diagnostic distance heuristics—remain swappable beneath stable interface contracts.

### 1.2 Verification Invariants

Every test gate enforces five fundamental invariants:
1. **Behavioral Invariance:** Forward evaluation results, output shapes, error codes, and exception types match specification contracts regardless of device (`cpu`, `cuda`) or precision mode.
2. **Mathematical & Numerical Parity:** All operator kernels produce outputs numerically close to mathematical reference models within strict floating-point tolerances (`rtol=1e-4`, `atol=1e-5` for float32/float64; `1e-3` for half-precision/Triton kernels), and satisfy exact analytical Jacobian checks under `gradcheck`.
3. **Autograd Gradient Flow:** Every trainable parameter in the built `GraphModule` receives valid, non-null, finite gradients during backward passes, ensuring absence of detached computation paths, dead parameters, or `NaN`/`inf` overflows.
4. **Compilation Compatibility:** Operators and network blocks run cleanly under `torch.compile(..., fullgraph=True)` with zero unexpected graph breaks and zero `FakeTensor` cache poisoning across eager/compile boundaries.
5. **Progressive Testability & Clean Tagging:** Tests for features pending implementation in downstream milestones (M1–M5) are fully authored and genuine, organized cleanly or tagged with clear skip-attributions, ensuring test suite integrity without facade implementations.

---

## 2. The 4-Tier Testing Methodology

The verification matrix for HNDL v0.10.0 is organized into four progressive tiers:

```
+-----------------------------------------------------------------------------------+
| Tier 4: Real-World Application Workloads                                          |
| (MMDiT Diffusion Transformer, Neural Audio Codec, Vision-Language CLIP Dual       |
|  Encoder, Autoregressive Hybrid Decoder with Cache, Sparse MoE Training Pipeline, |
|  Standalone Python Codegen Round-Trip)                                            |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Interactions (Pairwise Combinations)                        |
| (Mamba2 + DropPath, JointAttention + DyTanh, .hndl Macros + Weight Tying,         |
|  Profiler + Checkpointing, CLI + MoE Profile, Codegen + State Rebinding,          |
|  2D RoPE + Rank-5 + Where, Snake + Causal Conv1d/Deconv1d, Diffing + SOTA)        |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases (Limits, Extremes, Defect Triggers)               |
| (Empty/single tokens T=1, odd lengths, Rank-5 boundaries, reshape -1 edge cases,  |
|  zero-scale init, numeric stability alpha->0, extreme parameter counts, CLI errs) |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (Happy Paths in Isolation)                               |
| (Bugs B-01..B-10, 6 New Ops, SOTA Extensions, Rank-5, .hndl Language Features,    |
|  Extensibility & Codegen, Static Profiler, CLI Tooling, Lifecycle, DX Diagnostics)|
+-----------------------------------------------------------------------------------+
```

### 2.1 Tier 1: Feature Coverage (Happy Paths in Isolation)
- **Scope:** Verifies primary functional requirements for every v0.10.0 deliverable in isolation.
- **Coverage Areas:**
  - **Area 1: Bug Remediations (B-01 through B-10):**
    - B-01: `cast` operator with non-compute dtypes (`int64`, `bool`, `float64`) through `GraphModule`, no `E_RUNTIME` mismatch, no `_sink` buffer pollution.
    - B-02: RoPE / Attention / LatentAttention caches skip `FakeTensor` retention during `torch.compile` tracing; clear cache on `.to()`/`_apply()`.
    - B-03: `gated_delta_attention` in `float16` forward and backward produces 100% finite gradients with zero `NaN`s as $\alpha, \beta \to 0$.
    - B-04: `grn` in `float16` does not overflow on large spatial activations ($H \times W \ge 1024$); `sinusoidal_embed` in `float16` accumulates in float32.
    - B-05: `matmul._relation` rank inference on rank-4 without non-monotone rank guess.
    - B-06: `conv1d` with `causal=True, stride > 1` records length interval without premature `E_AMBIGUOUS`.
    - B-07: `Registry.pretrained_provider` with `contract="ids"` and `shape_fn` without monkey-patching or multi-output crash.
    - B-08: `dit_tiny.hndl` with proper `transpose(1, 2)` before `reshape` and zero dead parameters; `deepseek_moe_tiny.hndl` with `expert_kind="swiglu"`.
    - B-09: Multiple `_` discard targets in tuple unpacking (`a, _, _ = chunk(3)`) without duplicate identifier rejection.
    - B-10: `l2_norm` rejects `dim=0` (batch axis) and out-of-range dim at resolution time; `transformer_block` forwards `eps=eps` to `Attention`.
  - **Area 2: New Built-in Operators (6 Operators):**
    - `mamba2` (`ssd@1`): State Space Duality block, causal depthwise conv, semi-separable chunked recurrence, gated RMSNorm.
    - `joint_attention` (`mmdit_attention@1`): Dual-stream multimodal attention with modality-specific QKV, joint SDPA, and separate output projections.
    - `dy_tanh` (`dynamic_tanh@1`): Dynamic Tanh with learned scalar alpha and per-channel affine gamma, beta.
    - `drop_path` (`stochastic_depth@1`): Residual stochastic depth with survival scaling in train mode and identity in eval mode.
    - `where` (`where@1`): Ternary selection consuming boolean condition mask and two tensor branches with bidirectional shape inference.
    - `snake` (`snake@1`): Periodic activation $x + \frac{1}{\alpha + \epsilon}\sin^2(\alpha x)$ for neural audio synthesis.
  - **Area 3: Operator SOTA Extensions (`since="0.10.0"`):**
    - RoPE frequency scaling (`rope_scaling="yarn"`, `"linear"`, `"llama3"`) and 2D axial RoPE (`spatial_shape`).
    - `resblock` bottleneck (`kind="bottleneck"`, `activation="silu"`).
    - `transformer_block` option parity (`head_dim`, `window_size`, `logit_softcap`, `zero_centered`, `post_norm`).
    - `deconv1d` with `causal=True` and `channels_last=True`.
    - `adaln_zero` with `gate=False` (emitting single modulated output).
    - `moe` with `routed_scaling_factor=2.5`.
    - `conv` / `conv1d` with `policy="same3"` and `groups="depthwise"`.
  - **Area 4: Solver & Rank Extensions:**
    - Rank-5 tensor support across `SUPPORTED_RANKS = (2, 3, 4, 5)` in solver and operations (`[B, C, T, H, W]`).
    - `reshape(-1, ...)` at arbitrary axis positions (e.g. `reshape(-1, 1, 1)` on `[B, C] -> [B, C, 1, 1]`).
  - **Area 5: .hndl Language Features:**
    - Scalar constant bindings (`D = 128` in `.hndl`).
    - `def` macros (non-recursive AST-expanded reusable blocks).
    - Literal tuple loops (`for c in (64, 128, 256):`).
    - Python callable `with ops.scope("prefix"):` and `registry.subgraph(alias, plan)`.
  - **Area 6: Extensibility, Weight Tying & Codegen:**
    - Declarative weight tying (`tie={"weight": "tok_emb.weight"}`).
    - Post-build state mutation blessing (`with model.allow_state_mutation():` and `model.rebind_state()`).
    - Standalone PyTorch codegen (`plan.to_python_source()`).
    - External checkpoint loading (`hndl.torch.load_weights(model, path/dict, key_map=...)`).
    - Plugin entrypoint discovery (`importlib.metadata.entry_points(group="hndl.operators")`).
  - **Area 7: Static Profiling, Diffing & Visualization:**
    - `plan.profile(batch_size=1)` computing parameter counts (total & active), forward FLOPs, slot-recycled peak activation memory, KV/recurrent state bytes.
    - Active vs total parameters on MoE.
    - `plan.to_mermaid()` producing valid Mermaid flowchart text with typed shapes.
    - `plan.diff(other_plan)` producing `PlanDiff` with added/removed/modified nodes, shape deltas, param/FLOP deltas.
  - **Area 8: CLI Tooling:**
    - `hndl inspect <file.hndl>` (terminal output of resolved contracts, parameters, FLOPs).
    - `hndl mermaid <file.hndl> --out <file.mmd>` (exporting Mermaid diagram).
    - `hndl check <path>` (validating recipes and sidecars, exit code 0/1).
  - **Area 9: Training & Inference Lifecycle Primitives:**
    - `model.param_groups(weight_decay=0.01)` deterministically partitioning 2D weights from 1D biases, norms, embeddings, log-scales.
    - Selective activation checkpointing `build(plan, checkpoint=True | Sequence[str])` active during `train()` and bypassed during `eval()`.
    - Graph-aware depth-scaled initialization `build(plan, init_policy="depth_scaled")` scaling residual exit projections by $1/\sqrt{2 N_{\text{layers}}}$.
    - Stateful autoregressive `InferenceCache` for decoding.
  - **Area 10: IDE Autocomplete, Typing & Diagnostics:**
    - `OperatorNamespace.__dir__()` returning all registered operator aliases.
    - Populated `__doc__` and `__signature__` on `ops.<alias>`.
    - `capture.pyi` type stub generation and `--check`.
    - PEP 562 lazy import of `hndl.torch`.
    - Fuzzy suggestions using `difflib.get_close_matches` across operators, arguments, policies, variables, parameters, nodes.
    - `_Solver` shape conflict provenance tracking in `E_CONSTRAINT`.
    - Line number capture in `source={"line": ...}`.
    - Digit-ending loop naming separation (`pwconv1_0`).

### 2.2 Tier 2: Boundary & Corner Cases (Limits, Extremes, Defect Triggers)
- **Scope:** Exercises system behavior under extreme dimensions, zero values, non-standard divisibility, and error invalidation paths.
- **Test Scenarios:**
  - Empty or single-token sequence contexts ($T=1$, batch=1).
  - Odd sequence lengths with RoPE, 2D axial RoPE, sliding windows, and chunk boundaries.
  - Extreme dimensions (single-layer $N=1$ depth scaling, massive parameter counts).
  - Rank-5 boundary cases (5D pooling, 5D reshape, 5D slicing).
  - Reshape `-1` edge cases (single element dimension, negative axis at middle vs end vs front).
  - Divisibility constraint violations in new operators (e.g. Mamba2 `heads * head_dim != inner_dim`).
  - Dynamic Tanh numerical stability with zero or extreme initial $\alpha$.
  - DropPath boundary cases: `drop_prob=0.0` (pure identity) vs `drop_prob=1.0` (all dropped in train mode).
  - Where operator with mismatched branch shapes or invalid condition dtype (non-boolean).
  - Snake operator with $\alpha \to 0$ ($\epsilon$ numerical protection).
  - CLI with nonexistent files, corrupted `.hndl` syntax, invalid arguments.
  - Profiler with non-positive batch size, zero-length sequence.
  - `param_groups` on models with 0 parameters or all-1D parameters.
  - Depth-scaled initialization on models with 0 residual branches.
  - `InferenceCache` with mismatched cache keys or step dimensions.

### 2.3 Tier 3: Cross-Feature Interactions (Pairwise Combinations)
- **Scope:** Verifies that new and extended features interact seamlessly without unintended coupling, shape mismatch, or graph resolution conflicts.
- **Pairwise Interactions Tested:**
  1. `mamba2` + `drop_path` residual block.
  2. `joint_attention` + `dy_tanh` multimodal block with dynamic tanh normalization.
  3. `.hndl` macros with tuple loops and declarative weight tying.
  4. Static Profiler + Depth-Scaled Initialization + Selective Checkpointing on a multi-layer Transformer.
  5. CLI inspect + Static Profiling on a Sparse MoE network with active vs total parameter reporting.
  6. `InferenceCache` + `model.param_groups` + autoregressive decoding.
  7. Standalone Python codegen (`to_python_source`) + `allow_state_mutation()` + weight tying.
  8. Unrolled tuple loops + line number provenance + fuzzy error reporting.
  9. 2D Axial RoPE + Rank-5 Video tensor reshaping + `where` masking.
  10. `snake` periodic activation + causal `conv1d` + causal `deconv1d` audio codec block.
  11. `resblock` bottleneck (`kind="bottleneck"`) + `drop_path` + depth-scaled init.
  12. `where` boolean mask selection + `cast("bool")` + `l2_norm`.
  13. `adaln_zero` with `gate=False` + `dy_tanh` + `checkpoint=True`.
  14. `load_weights` with external state dict key mapping into a model with tied weights.
  15. Plan diffing (`plan.diff`) comparing baseline architecture vs architecture with `mamba2` and `drop_path`.

### 2.4 Tier 4: Real-World Application Workloads
- **Scope:** Validates end-to-end multi-layer architectures matching frontier production models with full forward and backward autograd gradient checks.
- **Frontier Workloads:**
  - **Workload 1: Multimodal Diffusion Transformer (MMDiT / Flux Style) Pipeline:**
    Dual-stream text and image processing using `joint_attention`, `adaln_zero`, `sinusoidal_embed`, selective checkpointing, and depth-scaled residual initialization, with complete forward evaluation and backward autograd gradient flow verification.
  - **Workload 2: Neural Audio Codec (EnCodec / DAC Style) Pipeline:**
    1D causal convolution frontend, causal transposed convolution backend (`deconv1d`), `snake` learned periodic activations, and `drop_path` regularization, ensuring finite gradients and exact waveform dimension preservation.
  - **Workload 3: Multimodal Vision-Language Dual Encoder (CLIP Style) Pipeline:**
    Vision tower with `resblock` (bottleneck) / ConvNeXt-V2 (`grn`), text tower with `attention` (QK-norm, RoPE), token pooling, `l2_norm`, and batched `matmul` cosine similarity matrix calculation.
  - **Workload 4: Autoregressive Hybrid Decoder with Stateful InferenceCache:**
    Hybrid sequence decoding combining `mamba2` state-space duality and `latent_attention` (MLA) with `InferenceCache` (prefill step $T=4$ followed by step-by-step decoding $T=1$), verifying state persistence and numerical equivalence.
  - **Workload 5: Sparse MoE Training Lifecycle Pipeline:**
    DeepSeek-V3 style sparse MoE with SwiGLU experts, sigmoid affinity routing, `model.param_groups(weight_decay=0.01)`, AdamW optimizer step, and selective activation checkpointing.
  - **Workload 6: Standalone Python Codegen & PyTorch Native Round-Trip:**
    Compiles a complete hybrid model to standalone PyTorch source code via `plan.to_python_source()`, dynamically executes the generated source in an isolated namespace, builds the native `torch.nn.Module`, loads weights from the HNDL model, and verifies identical forward outputs within $10^{-6}$ numerical tolerance.

---

## 3. Comprehensive Feature Inventory Coverage Matrix

The table below maps all v0.10.0 requirements and features to testing tiers, primary test classes, and verification strategies:

| # | Feature / Requirement | Tier | Test Class / Scope | Verification Strategy & Invariants |
|---|----------------------|------|--------------------|-----------------------------------|
| 1 | B-01: `cast` non-compute dtypes & buffer fix | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Verifies `cast("int64")`, `cast("bool")`, `cast("float64")` through `GraphModule`, checks no `_sink` in `state_dict()`. |
| 2 | B-02: RoPE / Attention compile cache fix | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Verifies zero `FakeTensor` retention during tracing and cache clearing on `_apply()`. |
| 3 | B-03: `gated_delta_attention` float16 backward | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Direct float32 log-alpha & recurrence accumulation, verifying 100% finite backward gradients. |
| 4 | B-04: `grn` & `sinusoidal_embed` float16 stability | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | float32 norm accumulation preventing overflow, high precision freqs/angles. |
| 5 | B-05: `matmul._relation` rank inference | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Rank-4 backward propagation without non-monotone rank 3 guess. |
| 6 | B-06: `conv1d` causal stride interval | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Records length interval for `stride > 1` without premature `E_AMBIGUOUS`. |
| 7 | B-07: `Registry.pretrained_provider` hardening | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Built-in `contract` and `shape_fn` on `Registry` without monkey-patching; multi-output support. |
| 8 | B-08: `dit_tiny` & `deepseek_moe_tiny` recipe fixes | Tier 1, 4 | `TestTier1FeatureCoverage`, `TestTier4RealWorldWorkloads` | Spatial transpose before reshape in `dit_tiny`, zero dead params, updated MoE args. |
| 9 | B-09: Multi-output `_` discard unpacking | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Allows `a, _, _ = chunk(3)` in `.hndl` and config interpreter. |
| 10 | B-10: `l2_norm` dim check & `transformer_block` eps | Tier 1, 2 | `TestTier1FeatureCoverage`, `TestTier2BoundaryCornerCases` | Rejects batch dim=0 at resolution time; forwards eps to attention; `extra_repr()` on 8 ops. |
| 11 | `mamba2` (`ssd@1`) Operator | Tier 1, 2, 3, 4 | All Tiers | State Space Duality block, causal conv1d, semi-separable recurrence, gated RMSNorm. |
| 12 | `joint_attention` (`mmdit_attention@1`) Operator | Tier 1, 2, 3, 4 | All Tiers | Dual-stream multimodal attention, modality QKV projs, joint SDPA, token split. |
| 13 | `dy_tanh` (`dynamic_tanh@1`) Operator | Tier 1, 2, 3 | Tier 1, 2, 3 | Dynamic Tanh with learned scalar alpha and per-channel affine gamma, beta. |
| 14 | `drop_path` (`stochastic_depth@1`) Operator | Tier 1, 2, 3, 4 | All Tiers | Stochastic depth survival scaling in training mode, exact identity in eval mode. |
| 15 | `where` (`where@1`) Operator | Tier 1, 2, 3 | Tier 1, 2, 3 | Ternary selection consuming boolean condition mask with bidirectional shape inference. |
| 16 | `snake` (`snake@1`) Operator | Tier 1, 2, 3, 4 | All Tiers | Periodic activation for audio synthesis with epsilon stability at alpha -> 0. |
| 17 | RoPE Scaling & 2D Axial RoPE | Tier 1, 2, 3 | Tier 1, 2, 3 | YaRN / Linear / Llama3 scaling on attention/MLA/rope, 2D grid coordinates. |
| 18 | `resblock` Bottleneck & Activation | Tier 1, 3, 4 | Tier 1, 3, 4 | `kind="bottleneck"`, `activation="silu"` / `"gelu"`, ResNet-50+ topology. |
| 19 | `transformer_block` Option Parity | Tier 1, 2 | Tier 1, 2 | `head_dim`, `window_size`, `logit_softcap`, `zero_centered`, `post_norm`. |
| 20 | `deconv1d` Causal & Channels-Last | Tier 1, 2, 3, 4 | All Tiers | `causal=True` left-padding, `channels_last=True` for sequence generation. |
| 21 | `adaln_zero` Gate Option | Tier 1, 2, 3, 4 | All Tiers | `gate=False` emitting single modulated output with 2D projection. |
| 22 | `moe` Routed Scaling Factor | Tier 1, 2, 3, 4 | All Tiers | `routed_scaling_factor` scaling active expert accumulations. |
| 23 | `conv` / `conv1d` same3 & depthwise | Tier 1, 2 | Tier 1, 2 | `policy="same3"` (k=3, s=1, p=1) and `groups="depthwise"`. |
| 24 | Rank-5 Tensor Support | Tier 1, 2, 3 | Tier 1, 2, 3 | 5D `[B, C, T, H, W]` support across solver, reshape, and permute. |
| 25 | `reshape(-1, ...)` Arbitrary Inferred Axis | Tier 1, 2, 3 | Tier 1, 2, 3 | Resolves -1 at any axis position (front, middle, end) based on total elements. |
| 26 | .hndl Scalar Constants | Tier 1, 2, 3 | Tier 1, 2, 3 | Immutable scalar bindings `D = 128` evaluated in AST interpreter. |
| 27 | .hndl `def` Macros | Tier 1, 2, 3 | Tier 1, 2, 3 | Non-recursive AST-expanded reusable blocks with automatic node scoping. |
| 28 | .hndl Literal Tuple Loops | Tier 1, 2, 3 | Tier 1, 2, 3 | Literal loops `for c in (64, 128, 256):` unrolled during config evaluation. |
| 29 | Python `ops.scope()` & Subgraph Registration | Tier 1, 3 | Tier 1, 3 | Scope context manager and `registry.subgraph(alias, plan)` composite ops. |
| 30 | Declarative Weight Tying | Tier 1, 2, 3, 4 | All Tiers | `tie={"weight": "tok_emb.weight"}` bound during build, recorded in snapshot. |
| 31 | Post-Build State Mutation Blessing | Tier 1, 2, 3, 4 | All Tiers | `with model.allow_state_mutation():` and `model.rebind_state()` for PEFT/quant. |
| 32 | Standalone PyTorch Codegen | Tier 1, 3, 4 | Tier 1, 3, 4 | `plan.to_python_source()` emitting zero-dependency `nn.Module` with shapes. |
| 33 | Checkpoint Loader (`load_weights`) | Tier 1, 3, 4 | Tier 1, 3, 4 | External checkpoint loader with key remapping and validation. |
| 34 | Entry-Point Plugin Discovery | Tier 1 | Tier 1 | Discovery of third-party operators via `importlib.metadata`. |
| 35 | Static Profiler (`plan.profile()`) | Tier 1, 2, 3 | Tier 1, 2, 3 | Allocation-free params (total & active), FLOPs, peak memory, KV/state bytes. |
| 36 | Visual Mermaid Export (`plan.to_mermaid()`) | Tier 1, 2 | Tier 1, 2 | Shape-annotated Mermaid flowchart export. |
| 37 | Semantic Plan Diffing (`plan.diff()`) | Tier 1, 3 | Tier 1, 3 | Structural comparison of nodes, shapes, parameter deltas, and FLOP deltas. |
| 38 | HNDL CLI Tooling (`hndl`) | Tier 1, 2, 3 | Tier 1, 2, 3 | `inspect`, `mermaid`, and `check` subcommands with return codes. |
| 39 | Optimizer Parameter Groups | Tier 1, 2, 3, 4 | All Tiers | `model.param_groups(weight_decay=0.01)` separating 2D weights from 1D params. |
| 40 | Selective Activation Checkpointing | Tier 1, 2, 3, 4 | All Tiers | `build(plan, checkpoint=...)` wrapping heavy blocks in non-reentrant checkpoint. |
| 41 | Graph-Aware Depth-Scaled Init | Tier 1, 2, 3, 4 | All Tiers | Scales residual exit projections by $1/\sqrt{2 N_{\text{layers}}}$. |
| 42 | Stateful `InferenceCache` | Tier 1, 2, 3, 4 | All Tiers | KV cache, MLA compressed cache, KDA matrix, Mamba2 state decoding. |
| 43 | IDE Autocomplete & Signatures | Tier 1 | Tier 1 | `ops.__dir__()`, `__doc__`, `__signature__` on operator namespaces. |
| 44 | Type Stub Generation (`capture.pyi`) | Tier 1 | Tier 1 | `python -m hndl.docs` generates and verifies `src/hndl/capture.pyi`. |
| 45 | PEP 562 Lazy Import | Tier 1 | Tier 1 | Lazy access to `hndl.torch` without eager torch import; `__all__` manifests. |
| 46 | Fuzzy Error Suggestions | Tier 1, 2, 3 | Tier 1, 2, 3 | `difflib` suggestions across operators, args, policies, variables, nodes. |
| 47 | Solver Shape Provenance Tracking | Tier 1, 2, 3 | Tier 1, 2, 3 | Tracking `(ref, axis)` origins in `_Solver` to pinpoint conflict sources. |
| 48 | Line Numbers & Loop Naming | Tier 1, 3 | Tier 1, 3 | Source line capture in `source={"line": ...}`; digit-ending loop separator. |

---

## 4. Verification Commands & Runner Instructions

The test suite is executable via standard `uv` entrypoints:

```bash
# 1. Lint and style check for E2E suite:
uv run ruff check tests/e2e/

# 2. Run the complete v0.10.0 E2E test suite:
uv run pytest tests/e2e/test_v010_e2e.py -v

# 3. Run individual testing tiers:
uv run pytest tests/e2e/test_v010_e2e.py -k "TestTier1" -v
uv run pytest tests/e2e/test_v010_e2e.py -k "TestTier2" -v
uv run pytest tests/e2e/test_v010_e2e.py -k "TestTier3" -v
uv run pytest tests/e2e/test_v010_e2e.py -k "TestTier4" -v

# 4. Run documentation generator check:
uv run python -m hndl.docs --check

# 5. Run full test suite:
uv run pytest -m "not benchmark"
```

---

## 5. Verification Thresholds & Acceptance Criteria

1. **Test Suite Integrity:**
   - Zero facade tests, zero hardcoded dummy results, zero tautological assertions (`assert True`).
   - Every test constructs authentic plans/models, executes forward/backward passes, checks exact shapes, finite gradients, or verified error codes.
2. **Linting & Code Quality:**
   - `uv run ruff check tests/e2e/` must exit with code 0 (zero errors, zero warnings).
3. **Execution Reliability:**
   - 100% of implemented features must pass without failure. Tests for features pending implementation in downstream milestones must report explicit, descriptive skip reasons.
