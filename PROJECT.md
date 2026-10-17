# Project: HNDL SOTA Operator Catalog, Pretrained Provider Contract Hardening, and Network Recipes

## Architecture
HNDL is structured across compiler, runtime, operator catalog, and verification layers:
- **Specification & Parsing** (`src/hndl/_parser.py`, `src/hndl/capture.py`, `src/hndl/ast.py`): Parses DSL, validates syntax, extracts node declarations and contracts.
- **Type System & Settings** (`src/hndl/types.py`, `src/hndl/settings.py`): Immutable plan types, dictionary input contracts (`named_dtypes`), port specifications, serialization.
- **Registry & Operator Specs** (`src/hndl/registry.py`, `src/hndl/operator.py`, `src/hndl/relations.py`): Catalog of 78 built-in operations, shape relations, `Arg(..., since="0.9.0")`.
- **Resolver & Constraint Solver** (`src/hndl/resolver.py`): Topological sort, bidirectional shape inference, `ELLIPSIS` axis matching, dtype propagation (including `cast`).
- **PyTorch Runtime** (`src/hndl/torch.py`): `GraphModule` execution engine, `_build_fast_program`, state-store tracking, dynamic shape validation, `torch.compile` compatibility.
- **Operator Catalog** (`src/hndl/operators/`):
  - Existing extended: `attention`, `cross_attention`, `transformer_block`, `moe`, `conv1d`, `rms_norm`, `layer_norm`, `adaptive_norm`, `matmul`.
  - New Tier 1: `gated_delta_attention` (KDA / Gated DeltaNet), `latent_attention` (MLA), `l2_norm`, `adaln_zero`, `rope`.
  - New Tier 2: `deconv1d` (`conv_transpose1d@1`), `grn`, `sinusoidal_embed`, `cast`.
- **Pretrained Subsystem** (`src/hndl/pretrained.py`): Provider registrations, shape/contract callbacks avoiding `meta` execution, `contract="ids"` token-ID inputs.
- **Authored Network Recipes** (`examples/networks/`): 21 total recipes (13 existing + 8 new SOTA: `llama_tiny`, `deepseek_moe_tiny`, `whisper_seq2seq_tiny`, `convnext_tiny`, `espcn_super_res`, `kimi_linear_tiny`, `qwen3_gqa_qknorm`, `dit_tiny`).
- **Verification, Benchmarks & Documentation** (`tests/`, `tests/operators/`, `tests/benchmark/`, `src/hndl/docs.py`, `SPEC.md`, `IMPLEMENTATION.md`, `CHANGELOG.md`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `attention` GQA/MQA | `kv_heads` with fused `enable_gqa=True` in `F.scaled_dot_product_attention` | M1 | R1, Survey |
| 2 | `attention` `head_dim` override | Independent `head_dim` override decoupling $H \cdot d_h$ from $D$ | M1 | R1, Survey |
| 3 | `attention` per-head QK normalization | `qk_norm` supporting `"none"`, `"rms_norm"`, `"l2_norm"` per-head before RoPE | M1 | R1, Survey |
| 4 | `attention` configurable `rope_theta` | Configurable base frequency replacing hardcoded 10000.0 | M1 | R1, Survey |
| 5 | `attention` sliding-window attention | `window_size` sliding window band masking | M1 | R1, Survey |
| 6 | `attention` `logit_softcap` | Explicit fallback path applying $C \cdot \tanh(\text{scores} / C)$ | M1 | R1, Survey |
| 7 | `cross_attention` SOTA extensions | Add `qkv_bias`, `out_bias`, `kv_heads`, `head_dim`, `qk_norm`, `logit_softcap` | M1 | R1, Survey |
| 8 | `transformer_block` FFN kind & hidden override | `ffn_kind` (`"mlp"` vs `"swiglu"`) and explicit inner `hidden` width override | M1 | R1, Survey |
| 9 | `transformer_block` RoPE & Attention options | Pass-through `rope`, `rope_theta`, `kv_heads`, `qk_norm`, and `bias` | M1 | R1, Survey |
| 10 | `moe` SwiGLU Gated Experts | 3-projection gated experts (`expert_kind="swiglu"` alongside `"mlp"`) | M1 | R1, Survey |
| 11 | `moe` Bias & Sigmoid Affinity | `bias` (bool, default True) and `scoring` (`"softmax"` vs `"sigmoid"`) | M1 | R1, Survey |
| 12 | `moe` Compile Compatibility & Reference Oracle | 0 graph breaks under `torch.compile(fullgraph=True)` and updated dense `_reference` oracle | M1 | R1, Survey |
| 13 | `conv1d` Causal & Channels-Last | `causal=True` (left padding, bijective $L_{out} == L_{in}$ in solver when stride=1) and `channels_last=True` | M1 | R1, Survey |
| 14 | `rms_norm` Zero-Centered Scaling | `zero_centered=True` (Gemma $1 + w$ scaling, weight initialized to 0.0) | M1 | R1, Survey |
| 15 | `layer_norm` Bias Option | `bias=False` support using `elementwise_affine` | M1 | R1, Survey |
| 16 | `adaptive_norm` Rank-3 Sequence Support | Rank-3 `[B, T, C]` sequence modulation alongside rank-4 `[B, C, H, W]` | M1 | R1, Survey |
| 17 | `matmul` Multi-Rank Batching | Generalize to rank 2, 3, and 4 batched matrix multiplication (`a[B, ..., N, K], b[B, ..., K, M] -> out[B, ..., N, M]`) | M1 | R1, Survey |
| 18 | `gated_delta_attention` Operator Definition | `gated_delta_attention@1` in category `"sequence"` with KDA (`decay_kind="channel"`) and Gated DeltaNet (`decay_kind="head"`) | M2 | R2, Survey |
| 19 | `gated_delta_attention` Input Projections | Causal short depthwise conv1d (`conv_size=4`), SiLU, per-head L2 Q/K norm, beta gate | M2 | R2, Survey |
| 20 | `gated_delta_attention` Output Projections | Head-wise RMSNorm + low-rank sigmoid output gate + output projection $W_o$ | M2 | R2, Survey |
| 21 | `gated_delta_attention` Chunkwise WY Forward | Pure PyTorch chunkwise parallel Tensor Core evaluation via `torch.linalg.solve_triangular` | M2 | R2, Survey |
| 22 | `gated_delta_attention` Sequential Recurrence Reference | Exact step-by-step state recurrence in `_reference(module)` | M2 | R2, Survey |
| 23 | `latent_attention` Operator Definition | `latent_attention@1` in category `"sequence"` implementing Multi-Head Latent Attention (MLA) | M2 | R2, Survey |
| 24 | `latent_attention` LoRA Compression & NoPE Mode | KV LoRA rank, optional Q LoRA rank, decoupled RoPE, and `qk_rope_head_dim=0` NoPE mode | M2 | R2, Survey |
| 25 | `l2_norm` Operator | `l2_norm@1` in category `"normalization"`: Parameter-free L2 norm with `dim` and `eps` | M2 | R2, Survey |
| 26 | `adaln_zero` Operator | `adaln_zero@1` in category `"normalization"`: Rank-3 adaptive LayerNorm-Zero returning `out` and `gate` | M2 | R2, Survey |
| 27 | `rope` Standalone Operator | `rope@1` in category `"sequence"`: Standalone rotary position embedding with `heads` and `theta` | M2 | R2, Survey |
| 28 | `deconv1d` Operator | `conv_transpose1d@1` in category `"convolution"` with `policy="up2"` using `conv_transpose_axis` | M2 | R2, Survey |
| 29 | `grn` Operator | `grn@1` in category `"normalization"`: Global Response Normalization for ConvNeXt-V2 | M2 | R2, Survey |
| 30 | `sinusoidal_embed` Operator | `sinusoidal_embed@1` in category `"sequence"`: Deterministic log-spaced sinusoidal timestep embeddings | M2 | R2, Survey |
| 31 | `cast` Operator | `cast@1` in category `"core"`: Explicit dtype conversion across edges with resolver integration | M2 | R2, Survey |
| 32 | `pretrained` Explicit Shape Callback | Optional `shape_fn` on `registry.pretrained_provider` avoiding `torch.device("meta")` execution | M3 | R3, Survey |
| 33 | `pretrained` Token-ID Input Support | Support `contract="ids"` (`int64`, rank 2 `[B, T]`) in `LocalProvider` and `pretrained` operator | M3 | R3, Survey |
| 34 | `pretrained` Hardening Test Suite | Unit & integration tests for Triton/CUDA kernel wrapping and token-ID inputs | M3 | R3, Survey |
| 35 | `llama_tiny` Network Recipe | 27 nodes, 918,656 parameters (`.hndl` and `.json` sidecars) | M4 | R4, Survey |
| 36 | `deepseek_moe_tiny` Network Recipe | 19 nodes, 922,240 parameters (`.hndl` and `.json` sidecars) | M4 | R4, Survey |
| 37 | `whisper_seq2seq_tiny` Network Recipe | 20 nodes, 159,296 parameters (multi-input `x` float32 + `tokens` int64) | M4 | R4, Survey |
| 38 | `convnext_tiny` Network Recipe | 19 nodes, 76,876 parameters (`learned_scale` per instantiation) | M4 | R4, Survey |
| 39 | `espcn_super_res` Network Recipe | 6 nodes, 26,796 parameters (sub-pixel conv with backward solver inference) | M4 | R4, Survey |
| 40 | `kimi_linear_tiny` Network Recipe | 3:1 hybrid KDA + NoPE MLA, 956,384 parameters | M4 | R4, Survey |
| 41 | `qwen3_gqa_qknorm` Network Recipe | GQA (4:2), QK-RMSNorm, high-theta RoPE, SwiGLU, 853,376 parameters | M4 | R4, Survey |
| 42 | `dit_tiny` Network Recipe | AdaLN-Zero diffusion transformer, 695,472 parameters | M4 | R4, Survey |
| 43 | Network Recipe Test Scaling | `tests/test_network_examples.py` scaling from 40 to 64 tests (exact params, JSON digest, CPU/CUDA grads) | M4 | R4, Survey |
| 44 | Global Operator Count Synchronization | Synchronize count (69 -> 78) across `SPEC.md`, `IMPLEMENTATION.md`, `README.md`, `tests/test_package.py`, `tests/e2e/test_audit_remediation_e2e.py` | M4 | R4, Survey |
| 45 | Documentation Generator Update | `uv run python -m hndl.docs` generating 79 operator pages and updated `docs/networks.md` passing `--check` | M4 | R4, Survey |
| 46 | Repository Hygiene & Changelog | Maintain `## Unreleased` at top of `CHANGELOG.md`, incremental commits and pushes on `develop` | M4 | R4, AGENTS.md |
| 47 | E2E Testing Suite (Tiers 1–4) | Requirement-driven opaque-box test cases across Tiers 1–4 publishing `TEST_READY.md` | E2E | Project Pattern |
| 48 | Final Milestone Phase 1: 100% E2E Pass | Run full E2E test suite across Tiers 1–4 until 100% passing | Final | Project Pattern |
| 49 | Final Milestone Phase 2: Adversarial Hardening | Tier 5 white-box challenger stress tests on edge cases and numerical limits | Final | Project Pattern |
| 50 | Final Gate Verification & Victory Audit | Run ruff, pytest, benchmarks, docs check, git push, handoff to Sentinel | Final | Project Pattern |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Existing Operator SOTA Extensions (`Arg(..., since="0.9.0")`) | Features 1–17 (`src/hndl/operators/{attention,cross_attention,transformer_block,moe,conv1d,rms_norm,layer_norm,adaptive_norm,matmul}.py`, unit tests) | none | PLANNED |
| M2 | New Tier 1 & Tier 2 Built-in Operators | Features 18–31 (`src/hndl/operators/{gated_delta_attention,latent_attention,l2_norm,adaln_zero,rope,deconv1d,grn,sinusoidal_embed,cast}.py`, `resolver.py`, unit tests) | none | PLANNED |
| M3 | `pretrained` Provider Contract Hardening | Features 32–34 (`src/hndl/pretrained.py`, `src/hndl/operators/pretrained.py`, `tests/test_pretrained.py`) | none | PLANNED |
| E2E | E2E Testing Track | Feature 47 (Requirement-driven test suite Tiers 1–4, `TEST_INFRA.md`, `TEST_READY.md`) | none | PLANNED |
| M4 | Authored SOTA Network Recipes & Repository Hygiene | Features 35–46 (`examples/networks/`, `tests/test_network_examples.py`, count sync in `SPEC.md`, `IMPLEMENTATION.md`, docs, `CHANGELOG.md`) | M1, M2, M3 | PLANNED |
| Final | Final Milestone: E2E Integration, Adversarial Hardening & Audit | Features 48–50 (Phase 1: 100% E2E pass, Phase 2: Tier 5 adversarial hardening, Victory Audit handoff) | M1, M2, M3, M4, E2E | PLANNED |

## Interface Contracts
### `src/hndl/operators/` (M1 & M2) ↔ `src/hndl/resolver.py` & `src/hndl/types.py` (M2 & Core)
- `cast@1`: `cast` operator uses `dtype` argument. `src/hndl/resolver.py:_propagate_dtypes` inspects `node.args["dtype"]` to propagate the casted dtype to downstream edges.
- `deconv1d`: Uses `hndl.relations.conv_transpose_axis` on axis 2 with `policy="up2"`.
- `Arg(..., since="0.9.0")`: Handled natively by `src/hndl/operator.py:omitted` and `src/hndl/torch.py:_construct` default rehydration.

### `src/hndl/pretrained.py` (M3) ↔ Network Recipes & Providers (M4 & External)
- `registry.pretrained_provider(name, shape_fn=..., contract=...)`:
  - If `shape_fn` is provided, `_traced_output_shapes` calls `shape_fn(args)` directly, bypassing `torch.device("meta")`.
  - If `contract="ids"`, `LocalProvider` and `pretrained` operator accept `int64` inputs of rank 2 (`[B, T]`).

### Network Recipes (M4) ↔ Operator Catalog (M1 & M2)
- `llama_tiny` & `deepseek_moe_tiny` & `convnext_tiny` & `whisper_seq2seq_tiny` & `espcn_super_res`: Ready today with existing catalog.
- `qwen3_gqa_qknorm`: Consumes M1 `kv_heads`, `qk_norm="rms_norm"`, `rope_theta`, `ffn_kind="swiglu"`.
- `kimi_linear_tiny`: Consumes M2 `gated_delta_attention` and `latent_attention` (with `qk_rope_head_dim=0`).
- `dit_tiny`: Consumes M2 `adaln_zero` and `sinusoidal_embed`.

### E2E Testing Track (E2E) ↔ Implementation Track (M1–M4, Final)
- `TEST_READY.md`: Created at project root by E2E track when test cases (Tiers 1–4) and runner are ready.
- All verification commands use standard entry points:
  - `uv run pytest`
  - `uv run pytest tests/benchmark/ -v -s`
  - `uv run python -m hndl.docs --check`
  - `uv run ruff check src tests examples scripts`

## Code Layout
- `src/hndl/operators/{attention,cross_attention,transformer_block,moe,conv1d,rms_norm,layer_norm,adaptive_norm,matmul}.py` — Owned exclusively by M1
- `tests/operators/test_{attention,cross_attention,transformer_block,moe,conv1d,rms_norm,layer_norm,adaptive_norm,matmul}.py` — Owned exclusively by M1
- `src/hndl/operators/{gated_delta_attention,latent_attention,l2_norm,adaln_zero,rope,deconv1d,grn,sinusoidal_embed,cast}.py` — Owned exclusively by M2
- `tests/operators/test_{gated_delta_attention,latent_attention,l2_norm,adaln_zero,rope,deconv1d,grn,sinusoidal_embed,cast}.py` — Owned exclusively by M2
- `src/hndl/resolver.py` (`_propagate_dtypes` cast branch) — Owned exclusively by M2
- `src/hndl/pretrained.py`, `src/hndl/operators/pretrained.py`, `tests/test_pretrained.py` — Owned exclusively by M3
- `examples/networks/*` — Owned exclusively by M4
- `tests/test_network_examples.py` — Owned exclusively by M4
- `SPEC.md`, `IMPLEMENTATION.md`, `README.md`, `CHANGELOG.md`, `tests/test_package.py`, `tests/e2e/test_audit_remediation_e2e.py` — Owned exclusively by M4
- `tests/e2e/test_sota_operators_e2e.py`, `TEST_INFRA.md`, `TEST_READY.md` — Owned exclusively by E2E
