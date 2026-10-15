# Project: HNDL v0.8.2 Optimization & Readability Release

## Architecture
HNDL is structured across compiler, runtime, operator catalog, and verification layers:
- **Specification & Parsing** (`src/hndl/_parser.py`, `src/hndl/capture.py`): Parses DSL, validates syntax, extracts node declarations and contracts.
- **Type System & Settings** (`src/hndl/types.py`, `src/hndl/settings.py`, `src/hndl/config.py`): Immutable plan types, node specifications, parameter settings, serialization.
- **Registry & Operator Specs** (`src/hndl/registry.py`, `src/hndl/operator.py`, `src/hndl/relations.py`): Catalog of operations, shape relations, contracts.
- **Resolver & Constraint Solver** (`src/hndl/resolver.py`): Topological sort, bidirectional shape inference, dimension bound enforcement.
- **PyTorch Runtime** (`src/hndl/torch.py`): `GraphModule` and `Network` execution engine, state-store tracking, dynamic shape validation, `torch.compile` compatibility.
- **Operator Catalog** (`src/hndl/operators/`): 69 built-in PyTorch operator implementations with custom autograd and reference implementations.
- **Pretrained & Tooling** (`src/hndl/pretrained.py`, `src/hndl/testing.py`, `src/hndl/docs.py`): Weight loading, testing utilities, doc generator.
- **Verification & Benchmarks** (`tests/`, `tests/operators/`, `tests/benchmark/`): Parity tests, autograd tests, latency benchmarks, packaging checks.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | C-Speed `_stores` Verification | Dual-tier store check in `GraphModule._execute` using native `id(t)` comparison, falling back to shape/dtype check only on ID divergence | M1 | R1, Plan 1.1 |
| 2 | Hot-Path Positional Tuple Binding | Return positional tuple in `_bind` for zero-dict allocations on standard `forward()` calls | M1 | R1, Plan 1.2 |
| 3 | 1-Input Signature Fast Path | Dedicated 5-tuple fast path in `_signature` avoiding list comprehension allocations | M1 | R1, Plan 1.2 |
| 4 | Multi-Input `itemgetter` | Use `operator.itemgetter` for multi-input nodes in `_build_fast_program` | M1 | R1, Plan 1.2 |
| 5 | Single-Output Direct Return | Allow `Network.forward` to bypass intermediate dictionary unpacking | M1 | R1, Plan 1.2 |
| 6 | Liveness Slot Recycling | Static liveness analysis & free-list recycling in `_build_fast_program` reducing peak intermediate slots by 50–84% | M1 | R1, Plan 1.3 |
| 7 | Non-Mutating Dynamo Tracing | Pass `cache=not compiling` to `_resolve_shapes` in `_run_checked` to avoid mutating `self._compiled` during tracing | M1 | R1, Plan 1.4 |
| 8 | Registration Hook Fast Path | Query `_watched.data.get(id(module))` in `_registration_hook` to bypass `WeakValueDictionary` `KeyError` | M1 | R1, Plan 3.2 |
| 9 | CPU `construct()` Fast Path | Call module constructor directly on CPU when `torch_function_stack` is empty, bypassing `torch.device("cpu")` overhead | M1 | R1, Plan 3.2 |
| 10 | Hoist `registry.activated()` | Hoist context manager outside node loops in `_build()` and `parameter_counts()` | M1 | R1, Plan 3.2 |
| 11 | Pretrained & Testing Cleanup | Bound `_file_digests` with `@lru_cache(256)`, unify safetensors loader, reuse plan in `testing.py` | M1 | R4, Plan 4.3 |
| 12 | Fused `RMSNorm.forward` | Use `F.rms_norm` in `RMSNorm.forward` and move explicit formula to `_reference` | M2 | R2, Plan 2.1 |
| 13 | Fused `AdaptiveNorm.forward` | Use `F.instance_norm` + `torch.addcmul` in `forward` and move explicit formula to `_reference` | M2 | R2, Plan 2.1 |
| 14 | RoPE & Causal Bias Optimization | Scalar base `ROPE_BASE ** -exp`, instance `_rope_cache` in `Attention`, and `torch.full(...).triu(1)` | M2 | R2, Plan 2.2 |
| 15 | `MixtureOfExperts` Fast & Dynamo Dispatch | Sparse `torch.where` + `index_add_` in eager mode; branchless dispatch when compiling to eliminate graph breaks | M2 | R2, Plan 2.3 |
| 16 | Separable 1D `AdaptiveAvgPool2d` | Replace nested 2D loops with separable 1D pooling $O(H+W)$, retaining `_explicit` in `_reference` | M2 | R2, Plan 2.4 |
| 17 | `EqualLinear` Precomputed Scale | Precompute `self.scale = in_features ** -0.5` in `__init__` | M2 | R2, Plan 2.5 |
| 18 | `Hopfield` Optimization | Hoist `stored_t = self.stored.t()` outside loop, prestore `(self.features,)` | M2 | R2, Plan 2.5 |
| 19 | `FeedForward` Direct Activations | Bind `F.gelu`, `F.relu`, `F.silu` directly without lambda wrappers | M2 | R2, Plan 2.5 |
| 20 | `PoolTokens` Bound Function | Bind pooling callable in `__init__` instead of per-call string check | M2 | R2, Plan 2.5 |
| 21 | Residual Scale-and-Add Fusion | Use `torch.addcmul(x, branch, gamma)` in `SpatialAttention` and `TransformerBlock` | M2 | R2, Plan 2.5 |
| 22 | Operator Relation Deduplication | Deduplicate `non_image`, `reduce_axis`, `conv_axis`, and `eval_weight` across operator catalog | M2 | R4, Plan 4.2 |
| 23 | Solver Per-Sweep Convergence | Early exit check (`if not self.changed: break`) after each sweep in `_Solver.run` (2x speedup) | M3 | R3, Plan 3.1 |
| 24 | Pre-Allocate `NodeView` & Metadata | Pre-allocate `NodeView` in `_Solver.__init__` and static `Operator` metadata in `Operator.__post_init__` | M3 | R3, Plan 3.1 |
| 25 | $O((V+E)\log V)$ Topo Sort | Use `heapq` and integer in-degrees in `_ordered_nodes` | M3 | R3, Plan 3.1 |
| 26 | Decompose `_ordered_nodes` | Decompose into 4 focused helpers (`_normalize_graph_nodes`, `_topological_sort`, `_validate_reachability`, `_propagate_dtypes`) | M3 | R4, Plan 4.1 |
| 27 | Solver Axis/Rank Fast Path & Limit Check | Early returns in `axis`/`rank`; enforce `max_elements` after merging axis updates in `set_shape` | M3 | R3, Plan 3.1 |
| 28 | Decompose `Capture.emit` | Split into `_bind_inputs` and `_rethrow_with_source`, use `_variadic_member` | M3 | R4, Plan 4.1 |
| 29 | Unify Plan Table & Shape Formatting | Extract `format_plan_table(plan, label)` and `format_shape(shape)` in `types.py` | M3 | R4, Plan 4.1 |
| 30 | Isolate `conv_axis` Unpacking | Extract `_unpack_conv_axis_args` in `relations.py` | M3 | R4, Plan 4.1 |
| 31 | Cache `semantic_digest` & Singletons | Cache `ResolvedPlan.semantic_digest` and use pre-frozen default settings singletons | M3 | R3, Plan 3.3 |
| 32 | Cache `Registry.builtins()` | Cache master registry builtins and skip redundant default validation in `normalize_arguments` | M3 | R3, Plan 3.3 |
| 33 | ASCII Fast Path & Parser Depth | Fast-path ASCII source location lookup and integrate AST depth validation in `Validator` | M3 | R3, Plan 3.3 |
| 34 | Redundant Re-Solve Skip in `_build()` | Stamp validated `ResolvedPlan` to skip re-running `validate_concrete_plan` in `_build()` | M3 | R3, Plan 3.2 |
| 35 | E2E Testing Infra & Benchmarks Consolidation | Consolidate `_random_inputs` into `tests/_network_examples.py`, add warmup passes, verify device printouts | E2E | R5, Plan 4.3 |
| 36 | Full Test & Benchmark Gate Verification | 100% pass across unit, operator, benchmark, doc tests on cpu and cuda:0 | M4 | R5, AC |
| 37 | Version Synchronization & Documentation | Update version 0.8.2 in `pyproject.toml`, `_version.py`, `README.md`, `SPEC.md`, `IMPLEMENTATION.md`, `tests/test_package.py`, and `CHANGELOG.md` | M4 | R5, AGENTS.md |
| 38 | Distribution Build, Git Tag & PyPI Release | Local wheel/sdist `twine check --strict`, tag `v0.8.2`, `gh release create v0.8.2`, and `.github/workflows/publish.yml` monitoring | M4 | R5, AC |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Runtime Forward-Pass & Memory Optimization | Features 1–11 (`src/hndl/torch.py`, `pretrained.py`, `testing.py`, `docs.py`) | none | DONE (commit 57d9515) |
| M2 | Operator Kernels & Forward-Pass Optimization | Features 12–22 (`src/hndl/operators/*.py`) | none | DONE (commit 45a6dec) |
| M3 | Compiler, Solver & Network Construction | Features 23–34 (`src/hndl/resolver.py`, `operator.py`, `types.py`, `relations.py`, `settings.py`, `registry.py`, `capture.py`, `_parser.py`) | none | DONE (commit 05175b1) |
| E2E | E2E Test Suite & Testing Infrastructure | Feature 35 (`tests/benchmark/*`, `tests/_network_examples.py`, `TEST_READY.md`) | none | DONE (TEST_READY.md published) |
| M4 | Final Milestone: Full Verification, Packaging & PyPI Release | Features 36–38 (All test gates, benchmark verification, version bump, tag, GitHub release, PyPI publish) | M1, M2, M3, E2E | IN_PROGRESS (efb6a915-adfa-4bba-974f-0bf7f0cd96ba) |

## Interface Contracts
### `src/hndl/torch.py` (M1) ↔ `src/hndl/types.py` & `src/hndl/resolver.py` (M3)
- `ResolvedPlan._validated_stamp: tuple[int, tuple]`: M3 stamps validated plans in `validate_concrete_plan` / `resolve_graph`. M1 checks this stamp in `_build()` to skip redundant validation.
- `format_plan_table(plan, label)`: M3 exports `format_plan_table` in `src/hndl/types.py`. M1 calls it in `GraphModule.__repr__`.
- `format_shape(shape)`: M3 exports `format_shape` in `src/hndl/types.py`. M1 and M3 call it for consistent tuple/str formatting.

### `src/hndl/operators/` (M2) ↔ `src/hndl/relations.py` (M3)
- `conv_axis` / `conv_output` / `conv_input_range`: M3 maintains function signature in `src/hndl/relations.py`. M2 consumes these helpers in `resblock.py` and `_pooling.py`.
- `non_image(s)`: Extracted in `src/hndl/relations.py` or `src/hndl/operators/_common.py` and consumed by `linear.py`, `feed_forward.py`, `swiglu.py`, `moe.py`, and `hopfield.py`.
- `eval_weight(module)`: Exported from `src/hndl/operators/_equalized.py` and consumed by `conv.py`, `conv1d.py`, `deconv.py`, and `linear.py`.

### E2E Testing & Verification (E2E / M4) ↔ Runtime & Operators (M1, M2, M3)
- `TEST_READY.md`: Created at project root by E2E track when benchmark consolidation and test runner contracts are ready.
- All test runs use entry points: `uv run pytest`, `uv run pytest tests/benchmark/ -v -s`, `uv run python -m hndl.docs --check ...`, `uv run ruff check src tests examples scripts`.

## Code Layout
- `src/hndl/torch.py` — Owned exclusively by M1
- `src/hndl/pretrained.py` — Owned exclusively by M1
- `src/hndl/testing.py` — Owned exclusively by M1
- `src/hndl/docs.py` — Owned exclusively by M1
- `src/hndl/operators/*.py` — Owned exclusively by M2
- `src/hndl/resolver.py` — Owned exclusively by M3
- `src/hndl/operator.py` — Owned exclusively by M3
- `src/hndl/types.py` — Owned exclusively by M3
- `src/hndl/relations.py` — Owned exclusively by M3
- `src/hndl/settings.py` — Owned exclusively by M3
- `src/hndl/registry.py` — Owned exclusively by M3
- `src/hndl/capture.py` — Owned exclusively by M3
- `src/hndl/_parser.py` — Owned exclusively by M3
- `tests/_network_examples.py` — Owned exclusively by E2E / M4
- `tests/benchmark/*` — Owned exclusively by E2E / M4
- `pyproject.toml`, `src/hndl/_version.py`, `CHANGELOG.md`, `README.md`, `SPEC.md`, `IMPLEMENTATION.md` — Owned exclusively by M4
