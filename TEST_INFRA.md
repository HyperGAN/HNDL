# HNDL Testing Infrastructure & Verification Architecture

**Version:** 0.8.2  
**Scope:** Verification Architecture, Test Suites, Benchmark Methodology, and Feature Coverage  
**Maintainer:** Worker E2E (Testing Infrastructure & Benchmark Engineer)  

---

## 1. Opaque-Box Testing Architecture

### 1.1 Philosophy & Core Principles

The HNDL test infrastructure is designed around the principle of **opaque-box behavioral verification**. Rather than inspecting internal micro-structures or patching private attributes, test suites interact strictly with HNDL through its authoritative public API boundaries:
- Declarative DSL & Specification: `hndl.resolve(source, **kwargs)`, `hndl.loads()`, `hndl.dump()`
- PyTorch Runtime & Execution: `hndl.torch.build(plan, device=...)`, `hndl.torch.network(source, ...)`
- Type Contracts & Registries: `hndl.Registry`, `hndl.Operator`, `hndl.ResolvedPlan`, `hndl.Graph`
- Pretrained & Model Tooling: `hndl.pretrained.load()`, `hndl.docs`, `hndl.testing`

This opaque-box stance ensures that tests verify genuine behavior, mathematical invariants, memory safety, and wall-clock execution speed, rather than implementation accidents. Internal architectural enhancements—such as C-speed native `_stores` pointer checks, static activation slot recycling, fused C++/CUDA operator kernels, and heap-based topological sorting—remain completely interchangeable behind stable interface contracts.

### 1.2 Verification Invariants

Every test gate enforces four fundamental invariants:
1. **Behavioral Invariance**: High-level execution outputs, forward evaluation results, error codes, and exception types match specification contracts regardless of optimization paths.
2. **Numerical & Autograd Equivalence**: All operator kernels produce outputs numerically close to mathematical reference models within strict floating-point tolerances (`rtol=1e-4`, `atol=1e-5` for float32; `1e-3` for CUDA Triton kernels), and satisfy exact analytical Jacobian checks under `gradcheck` and `gradgradcheck`.
3. **Execution Safety & State Integrity**: Network execution preserves PyTorch state encapsulation: parameters, buffers, and submodules can be mutated, cloned, or transferred across devices, with immediate detection on desynchronization.
4. **Compilation Transparency**: Built `GraphModule` instances act as standard `torch.nn.Module` objects, supporting full-graph TorchDynamo tracing (`torch.compile`) with zero unexpected graph breaks.

---

## 2. The 4-Tier Testing Methodology

The HNDL verification matrix is structured into four distinct, progressive tiers:

```
+-----------------------------------------------------------------------------------+
| Tier 4: Benchmark, Performance Parity & Compilation Gate                          |
| (tests/benchmark/ - Parity vs Handwritten PyTorch, Latency Catalog, torch.compile) |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 3: End-to-End Network Topology & Documentation Verification                  |
| (tests/test_network_examples.py, tests/test_package.py, hndl.docs)                |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 2: Operator Numerical Correctness & Autograd Gate                            |
| (tests/operators/ - Parity, First-Order gradcheck, Second-Order gradgradcheck)     |
+-----------------------------------------------------------------------------------+
                                          ▲
+-----------------------------------------------------------------------------------+
| Tier 1: Core Compiler, Types & DSL Specification Suite                            |
| (tests/test_*.py - Parser, Solver, Types, Registry, Capture, Config, Settings)   |
+-----------------------------------------------------------------------------------+
```

### 2.1 Tier 1: Core Compiler, Types & DSL Specification Suite

**Location:** `tests/test_*.py` (e.g., `test_parser.py`, `test_resolver.py`, `test_types.py`, `test_registry.py`, `test_capture.py`, `test_torch.py`)  
**Scope:** Unit-level verification of language parsing, abstract syntax trees, constraint propagation, type systems, and graph construction.

- **Grammar & Parsing**:
  - Validates full DSL syntax, nested arguments, positional and keyword port bindings.
  - Verifies AST depth bounds (guarding against recursion overflows) and precise line/column reporting.
  - Confirms fast-path ASCII column lookups produce identical error locations as Unicode fallback routines.
- **Bidirectional Constraint Solver**:
  - Exercises forward shape propagation, backward constraint inference, and channel dimension binding.
  - Verifies per-sweep fixed-point convergence (`self.changed` checks) reducing iteration count to $2N$.
  - Confirms early exits in `axis()` and `rank()`, and limit enforcement (`max_nodes`, `max_elements`, `max_iterations`).
  - Tests $O((V+E)\log V)$ topological sorting with cycle detection (`E_BINDING`).
- **Plan Immutability & Digests**:
  - Verifies `ResolvedPlan` immutability and JSON serialization/deserialization round-trips.
  - Confirms SHA-256 `semantic_digest` memoization produces bit-exact digests across executions.

### 2.2 Tier 2: Operator Numerical Correctness & Autograd Gate

**Location:** `tests/operators/`  
**Scope:** Exhaustive numerical verification of all 69 built-in PyTorch operators in the HNDL catalog.

Every operator is subjected to a three-stage mathematical gate:
1. **Reference Parity (`test_parity`)**: Compares operator output against an explicit, non-optimized reference model (`_reference.py`) across diverse tensor shapes, batch sizes, and data types (float32, float64, float16, bfloat16).
2. **First-Order Gradient Verification (`gradcheck`)**: Runs PyTorch's numerical differentiation engine `torch.autograd.gradcheck` in double precision (`float64`) with small finite-difference perturbations to verify analytical backward passes.
3. **Second-Order Gradient Verification (`gradgradcheck`)**: Runs `torch.autograd.gradgradcheck` to verify second derivatives, ensuring compatibility with higher-order optimizers, Hessian-vector products, and penalty losses.
4. **Edge Cases**: Zero-sized batches, non-contiguous strides, identity projections, dynamic spatial dimensions, and boundary values ($\epsilon$, extreme scales).

### 2.3 Tier 3: End-to-End Network Topology & Documentation Verification

**Location:** `tests/test_network_examples.py`, `tests/test_package.py`, `hndl.docs`  
**Scope:** Full-pipeline integration testing across authored neural network architectures.

- **13 Canonical Architectures** in `examples/networks/`:
  - `mlp`: Multi-layer perceptron with dense linear layers and activations.
  - `lenet5`: Classic convolutional vision network.
  - `resnet18`: Deep residual network with skip connections and batch normalization.
  - `vit_tiny`: Vision Transformer with patch embeddings and self-attention.
  - `gpt_tiny`: Causal autoregressive language model with learned embeddings and causal masking.
  - `moe_transformer`: Mixture-of-Experts transformer block with sparse gating.
  - `unet_small`: Encoder-decoder architecture with contraction, expansion, and cross-concatenation.
  - `autoencoder`: Convolutional bottleneck encoder-decoder.
  - `sagan_attention`: Self-attention generative adversarial trunk.
  - `hopfield_classifier`: Modern Hopfield associative memory network.
  - `dcgan_generator` & `dcgan_discriminator`: Transposed and strided convolutional GAN networks.
  - `conditional_discriminator`: Multi-input, multi-output conditional GAN discriminator.
- **Verification Properties**:
  - Spec sidecar validation: JSON metadata presence, schema adherence, and analytical parameter counts.
  - Construction & execution: Resolves and builds models on both CPU and CUDA with fixed initialization seeds.
  - Numerical sanity: Tensors remain finite across forward passes with verified output shapes.
  - Backward autograd flow: Verifies gradient accumulation across all trainable parameters.
  - Package integrity (`test_package.py`): Verifies distribution metadata, license inclusion, and entry points.

### 2.4 Tier 4: Benchmark, Performance Parity & Compilation Gate

**Location:** `tests/benchmark/`  
**Scope:** Wall-clock latency measurement, overhead bounding, and compiler acceleration checks.

- **Measurement Infrastructure (`_timing.py`, `conftest.py`)**:
  - Uses `torch.utils.benchmark.Timer` with blocked autoranging to eliminate timer resolution skew.
  - Enforces explicit CUDA device synchronization (`torch.cuda.synchronize()`) before and after timing runs.
  - Parametrized over all available hardware devices (`cpu`, `cuda:0`) with clear `[{device}]` label attribution.
  - Session-scoped configurable measurement budget via `HNDL_BENCHMARK_MIN_TIME` (default: 0.2s).
- **Parity vs. Hand-Written PyTorch (`test_parity_vs_handwritten_pytorch.py`)**:
  - Compares resolved HNDL `Network` against idiomatic hand-written PyTorch modules (`mlp`, `conv_stack`, `transformer_block`).
  - Warm-up pass executed prior to timing loop to eliminate first-pass caching/JIT skew.
  - Times 10 interleaved rounds alternating evaluation order to mitigate system load variance.
  - Enforces strict tolerance bounds: $\le 1.50\times$ for overhead-sensitive networks, $\le 2.00\times$ for arithmetic-dominated trunks.
- **Inference Latency Catalog (`test_example_networks_latency.py`)**:
  - Instruments per-inference execution times across all 13 canonical networks using consolidated `random_inputs`.
  - Verifies output tensor finiteness and shape contracts under benchmarking loads.
- **TorchDynamo & Inductor Compatibility (`test_compile_compat.py`)**:
  - Traces built networks with `torch.compile` and validates eager-vs-compiled numerical closeness.
  - Measures compilation speedups across CPU and CUDA, tracking compatibility without breaking on unsupported vendor kernels.

---

## 3. Feature Coverage Matrix

The table below maps all 38 features from `PROJECT.md § Feature Inventory` directly to their testing tiers, primary test files, and verification strategies:

| # | Feature Name | Tier | Primary Test File(s) | Verification Strategy & Invariants |
|---|--------------|------|----------------------|-----------------------------------|
| 1 | C-Speed `_stores` Verification | Tier 1, 4 | `tests/test_torch.py`, `tests/benchmark/test_parity_vs_handwritten_pytorch.py` | Native `id(t)` comparison fast path; confirms `< 3 µs` per-call overhead on `mlp` and `< 10 µs` on `resnet18`. |
| 2 | Hot-Path Positional Tuple Binding | Tier 1, 4 | `tests/test_torch.py`, `tests/benchmark/test_parity_vs_handwritten_pytorch.py` | Verifies zero dictionary allocations on positional `forward()` calls. |
| 3 | 1-Input Signature Fast Path | Tier 1 | `tests/test_torch.py` | Direct 5-tuple fast path avoiding list comprehension allocations for single-input graphs. |
| 4 | Multi-Input `itemgetter` | Tier 1 | `tests/test_torch.py`, `tests/test_network_examples.py` | Verifies multi-input node dispatch via `operator.itemgetter` in `_build_fast_program`. |
| 5 | Single-Output Direct Return | Tier 1 | `tests/test_torch.py` | Ensures single-output graphs return tensor directly without intermediate dict unpacking. |
| 6 | Liveness Slot Recycling | Tier 1 | `tests/test_torch.py` | Static liveness analysis verified; peak intermediate slots reduced by 50–84% without tensor overwrites. |
| 7 | Non-Mutating Dynamo Tracing | Tier 1, 4 | `tests/test_torch.py`, `tests/benchmark/test_compile_compat.py` | `cache=not compiling` prevents `self._compiled` mutation during `torch.compile` tracing passes. |
| 8 | Registration Hook Fast Path | Tier 1 | `tests/test_torch.py` | Fast-path lookup in `_watched.data.get(id(module))` bypassing `WeakValueDictionary` `KeyError`. |
| 9 | CPU `construct()` Fast Path | Tier 1 | `tests/test_torch.py` | Verifies direct constructor call bypassing `torch.device("cpu")` overhead when `torch_function_stack` is empty. |
| 10 | Hoist `registry.activated()` | Tier 1 | `tests/test_torch.py`, `tests/test_registry.py` | Validates context manager hoisting outside loops in `_build()` and `parameter_counts()`. |
| 11 | Pretrained & Testing Cleanup | Tier 1, 3 | `tests/test_pretrained.py`, `tests/test_testing.py` | `@lru_cache(256)` on `_file_digests`, unified safetensors loading, and plan reuse in test assertions. |
| 12 | Fused `RMSNorm.forward` | Tier 2 | `tests/operators/test_rms_norm.py` | Parity against explicit reference; verifies `F.rms_norm` speed and numerical accuracy. |
| 13 | Fused `AdaptiveNorm.forward` | Tier 2 | `tests/operators/test_adaptive_norm.py` | Fused `F.instance_norm` + `torch.addcmul` parity and gradient check. |
| 14 | RoPE & Causal Bias Optimization | Tier 2 | `tests/operators/test_attention.py` | Scalar base exponentiation, instance `_rope_cache` validation, and causal upper-triangular masking. |
| 15 | `MixtureOfExperts` Fast & Dynamo Dispatch | Tier 2, 4 | `tests/operators/test_moe.py`, `tests/benchmark/test_compile_compat.py` | Sparse `torch.where` + `index_add_` in eager; zero graph breaks under `torch.compile(..., fullgraph=True)`. |
| 16 | Separable 1D `AdaptiveAvgPool2d` | Tier 2 | `tests/operators/test_adaptive_avg_pool2d.py` | Parity between $O(H+W)$ separable pooling and explicit reference implementation. |
| 17 | `EqualLinear` Precomputed Scale | Tier 2 | `tests/operators/test_equal_linear.py` | Precomputed `in_features ** -0.5` scale verified for parity and numerical stability. |
| 18 | `Hopfield` Optimization | Tier 2 | `tests/operators/test_hopfield.py` | Parity and gradient verification with hoisted transpose operations. |
| 19 | `FeedForward` Direct Activations | Tier 2 | `tests/operators/test_feed_forward.py` | Direct function binding of `F.gelu`, `F.relu`, `F.silu` without lambda overhead. |
| 20 | `PoolTokens` Bound Function | Tier 2 | `tests/operators/test_pool_tokens.py` | Callable binding in `__init__` verified for pooling operators. |
| 21 | Residual Scale-and-Add Fusion | Tier 2 | `tests/operators/test_spatial_attention.py`, `tests/operators/test_transformer_block.py` | Fused `torch.addcmul(x, branch, gamma)` verified for exact parity. |
| 22 | Operator Relation Deduplication | Tier 1, 2 | `tests/test_relations.py`, `tests/operators/test_*.py` | Unified `non_image`, `reduce_axis`, `conv_axis`, and `eval_weight` across all operator specs. |
| 23 | Solver Per-Sweep Convergence | Tier 1 | `tests/test_resolver.py` | Convergence check after each sweep reduces solver iterations to $2N$ on forward-resolved graphs. |
| 24 | Pre-Allocate `NodeView` & Metadata | Tier 1 | `tests/test_resolver.py`, `tests/test_operator.py` | Pre-allocated `NodeView` and immutable `Operator` metadata verified for resolution correctness. |
| 25 | $O((V+E)\log V)$ Topo Sort | Tier 1 | `tests/test_resolver.py` | Heap-based topological sort verified on large DAGs, diamond graphs, and cycles (`E_BINDING`). |
| 26 | Decompose `_ordered_nodes` | Tier 1 | `tests/test_resolver.py` | 4 cohesive sub-functions tested independently for validation, ordering, reachability, and dtypes. |
| 27 | Solver Axis/Rank Fast Path & Limit Check | Tier 1 | `tests/test_resolver.py` | Early returns on matching rank/axis; post-merge `max_elements` resource limit enforcement. |
| 28 | Decompose `Capture.emit` | Tier 1 | `tests/test_capture.py` | Decomposed input binding and error rethrowing verified across AST patterns. |
| 29 | Unify Plan Table & Shape Formatting | Tier 1 | `tests/test_types.py` | `format_plan_table` and `format_shape` verified for string formatting uniformity. |
| 30 | Isolate `conv_axis` Unpacking | Tier 1 | `tests/test_relations.py` | `_unpack_conv_axis_args` isolated and verified across convolution and pooling relations. |
| 31 | Cache `semantic_digest` & Singletons | Tier 1 | `tests/test_types.py`, `tests/test_settings.py` | Memoized SHA-256 digest on `ResolvedPlan` and pre-frozen default settings singletons. |
| 32 | Cache `Registry.builtins()` | Tier 1 | `tests/test_registry.py` | Registry built-in caching verified; fast argument normalization without redundant validation. |
| 33 | ASCII Fast Path & Parser Depth | Tier 1 | `tests/test_parser.py` | ASCII index lookup and AST depth recursion limits verified on deep expressions. |
| 34 | Redundant Re-Solve Skip in `_build()` | Tier 1 | `tests/test_torch.py`, `tests/test_resolver.py` | Validated plan stamp check verified; skips redundant `validate_concrete_plan` calls in `_build()`. |
| 35 | E2E Testing Infra & Benchmarks Consolidation | Tier 4 | `tests/_network_examples.py`, `tests/benchmark/*` | Consolidated `random_inputs(plan, device, batch_size)`, pre-measurement warmups, and device tags. |
| 36 | Full Test & Benchmark Gate Verification | All Tiers | All suites | 100% pass across all unit, operator, packaging, and benchmark tests on CPU and CUDA. |
| 37 | Version Synchronization & Documentation | Tier 3 | `tests/test_package.py`, `docs` | Version `0.8.2` synchronized across pyproject, version file, documentation, and CHANGELOG. |
| 38 | Distribution Build, Git Tag & PyPI Release | Tier 3 | Release workflow | Clean dist build (`sdist`, `wheel`), `twine check --strict`, git tag `v0.8.2`, and PyPI publishing. |

---

## 4. Benchmark Utilities & Shared Helpers

### 4.1 Consolidated Input Generator (`tests/_network_examples.py`)

All benchmark and network integration suites generate test tensors through the consolidated `random_inputs` function:

```python
def random_inputs(
    plan: Any,
    device: str | torch.device = "cpu",
    batch_size: int = 2,
    generator: torch.Generator | None = None,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    ...
```

**Key Features:**
- **Flexible Invocation**: Accepts positional or keyword device and batch size, handling optional `torch.Generator` and `requires_grad` flags.
- **Type Safety**: Automatically matches each port's declared dtype (`float32`, `float64`, `float16`, `bfloat16`, `int64`, `int32`, `bool`), using bounded random integers for index inputs (e.g. token embeddings) and normal distributions for continuous activations.
- **Device & Generator Alignment**: Automatically guards against device mismatches between RNG generators and destination tensor memory.

### 4.2 Benchmark Instrumentation Rules

1. **Explicit Pre-Measurement Warmup**:
   Every latency comparison must execute a warmup pass (`hndl_module(*inputs); handwritten(*inputs)`) prior to timing. This ensures one-time compilation tasks (input signature validation, fast-program generation, CUDA context creation, kernel compilation) do not contaminate steady-state measurements.
2. **CUDA Synchronization**:
   On CUDA-enabled hardware, explicit calls to `torch.cuda.synchronize(device=device)` must surround timing windows to ensure asynchronous GPU kernel queues drain completely.
3. **Interleaved Timing Rounds**:
   When comparing alternative implementations, measurements must alternate execution order across at least 10 rounds to insulate results from external CPU/GPU throttling and host process load spikes.
4. **Device-Attributed Metric Logging**:
   All benchmark outputs must include explicit device tags (`[{device}]`) in printed lines to guarantee distinct, unambiguous performance tracking across CPU and CUDA backends.

---

## 5. Verification Commands & Entry Points

| Purpose | Command | Success Criteria |
|---------|---------|------------------|
| Unit & Integration Suite | `uv run pytest -m "not network and not benchmark"` | 100% pass, 0 failures |
| Operator Catalog Autograd Suite | `uv run pytest tests/operators/` | 100% pass across parity, gradcheck, and gradgradcheck |
| Example Network Pipeline | `uv run pytest tests/test_network_examples.py` | 100% pass across all 13 canonical architectures |
| Parity & Latency Benchmarks | `uv run pytest tests/benchmark/ -v -s` | Ratios $\le$ tolerances, finite metrics on CPU & CUDA |
| Documentation Consistency | `uv run python -m hndl.docs --check --out docs/operators --networks examples/networks --networks-out docs/networks.md` | Exit code 0, no uncommitted diffs |
| Code Formatting & Style | `uv run ruff check src tests examples scripts` | Clean check, 0 warnings/errors |
| Distribution Packaging Check | `uv run python -m build && uv run twine check --strict dist/*` | Exit code 0, package metadata clean |
