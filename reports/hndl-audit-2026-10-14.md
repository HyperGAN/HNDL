# HNDL Technical Audit Report

- **Date:** 2026-10-14
- **Project:** HNDL (*Human-readable Network Definition Language*)
- **Version Audited:** `0.8.0` (branch `develop`, commit `b70b99a`)
- **Platform & Runtime:** Linux, Python 3.11–3.14, PyTorch `>=2.6,<3`

---

## 1. Executive Summary

HNDL is a remarkably well-engineered library for defining neural network architectures with bidirectional shape inference, explicit input/output contracts, deterministic canonical plan serialization, and a low-overhead PyTorch execution backend. Its architecture cleanly separates the **torch-free symbolic core** (`_parser.py`, `config.py`, `capture.py`, `operator.py`, `registry.py`, `relations.py`, `resolver.py`, `settings.py`, `initializers.py`, `types.py`, `errors.py`) from the **PyTorch execution and checkpoint backends** (`torch.py`, `pretrained.py`, `operators/*.py`, `testing.py`).

### 1.1 Quantitative Codebase Profile

| Subsystem | Files | Lines | Key Responsibilities |
| :--- | ---: | ---: | :--- |
| **Core Engine (`src/hndl/*.py`)** | 17 | 5,916 | Lexer/AST config parser, symbolic capture, `@operator` schema, constraint solver, JSON plan persistence, PyTorch `GraphModule`/`Network` runtime, `pretrained` loader, doc generator, test harness |
| **Operator Catalog (`src/hndl/operators/*.py`)** | 73 | 5,863 | 69 public `@operator` modules across 12 categories + 4 internal helper modules (`__init__.py`, `_equalized.py`, `_pooling.py`, `_relations.py`) |
| **Unit & Integration Tests (`tests/`)** | 85 | 13,701 | Core engine tests (`tests/test_*.py`) and operator tests (`tests/operators/test_*.py`) |
| **Benchmarks (`tests/benchmark/`)** | 6 | 490 | Forward latency parity vs. handwritten PyTorch, example network latency, `torch.compile` compatibility |
| **Documentation & Specs (`*.md`, `docs/`)** | 80 | 11,609 | `SPEC.md`, `IMPLEMENTATION.md`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, guides, and 70 generated operator/network pages |
| **Examples & Build Scripts (`examples/`, `scripts/`)** | 28 | 571 | 13 `.hndl`/`.json` example networks, `adaptive_normalization.py`, MkDocs staging script |
| **Total Tracked** | **289** | **38,150** | |

### 1.2 Baseline Verification Results

- **Unit & Integration Test Suite (`pytest -m "not network and not benchmark"`):** `3,702 passed, 80 skipped, 53 deselected` in ~38s on CPU.
- **Linter (`ruff check src tests examples scripts`):** `All checks passed!`
- **Generated Documentation (`python -m hndl.docs --check`):** `70 operator pages and docs/networks.md are current.`

### 1.3 Architectural Strengths

1. **Zero-Execution Declarative Security Boundary (`src/hndl/_parser.py`, `src/hndl/config.py`):** Declarative `.hndl` configs are screened at the token stream level to prevent CPython 3.11–3.13 `ast.parse` C-stack overflows on small thread stacks, validated against a strict AST allowlist without `eval`/`exec`, and unrolled deterministically with pre-checked node budgets.
2. **Monotone Bidirectional Constraint Solver (`src/hndl/resolver.py`, `src/hndl/relations.py`):** Shape patterns (`x[B, ..., D] -> out[B, ..., 2*D]`) and custom relation functions propagate dimension facts forward and backward to a fixed point without guessing ambiguous extents, reporting interval bounds when strided convolutions admit multiple inverse sizes.
3. **Canonical Plan Persistence & Evolution (`src/hndl/types.py`, `src/hndl/operator.py`):** `ResolvedPlan` separates semantic digests (architecture, shapes, dtypes, initialization, trainability) from artifact digests (source locations, frontend provenance), and supports backward-compatible operator evolution via `Arg(since="...")` and deterministic plan completion on load.
4. **Once-Per-Signature Runtime Validation (`src/hndl/torch.py`):** `GraphModule` validates all intermediate tensor ports and module state on the first call for a given `(training, autocast, input_shapes, dtypes, devices)` signature, then executes subsequent calls on a slot-indexed fast path while still catching hookless `_parameters`/`_buffers`/`_modules` mutations in C (~0.7–1.3 µs overhead).
5. **Single-Source-of-Truth `@operator` Declarations (`src/hndl/operator.py`, `src/hndl/testing.py`, `src/hndl/docs.py`):** Every operator declaration drives argument normalization, shape solving, module construction, Markdown documentation generation, and the public `hndl.testing.check_operator` verification harness.

---

### 1.4 Summary of Audit Findings

The audit identified **32 concrete findings** (**8 High**, **14 Medium**, **10 Low**):

| ID | Severity | Component | Location | Summary |
| :--- | :--- | :--- | :--- | :--- |
| **H-01** | **High** | Parser | `src/hndl/_parser.py:156, 364-366` | Unhandled `IndexError` when parsing declarative source containing `\r` carriage returns (violates closed `HNDLError` contract). |
| **H-02** | **High** | Parser | `src/hndl/_parser.py:60, 86-91, 136-137` | Token screener (`_screen`) does not reset `previous` on `tokenize.NEWLINE`, falsely counting `(a, b) = ...` or `[a, b] = ...` unpacking after a call line as a chained operator and failing with `E_SYNTAX` at 33 lines. |
| **H-03** | **High** | Torch Backend | `src/hndl/torch.py:178-189, 883-885` | `_is_chain(plan)` treats single-output tuple-returning nodes (`chunk(1)`, `pretrained(..., layers=("...",))`) as unary chains (`_chain = True`), causing `model[:2](x)` to crash with `TypeError`; slices also accept `step != 1`. |
| **H-04** | **High** | Torch / Pretrained | `src/hndl/torch.py:1201-1216`, `src/hndl/operators/pretrained.py:208-216` | `pretrained(..., trainable=True)` leaves `layer.model.training == False` (`eval` mode) after `build()` / `network()`, violating `SPEC.md` and `IMPLEMENTATION.md`. |
| **H-05** | **High** | Pretrained | `src/hndl/pretrained.py:344-345` | `TimmProvider.instantiate` drops `self.source.revision` when loading `hf://owner/repo@revision`, pulling weights from the default branch instead of the pinned revision. |
| **H-06** | **High** | Operators (`rms_norm`) | `src/hndl/operators/rms_norm.py:76-80` | `RMSNorm.forward` unconditionally casts inputs via `x.float()`, truncating `float64` (`.double()`) precision to `float32`. |
| **H-07** | **High** | Packaging | `MANIFEST.in:1-2` | `MANIFEST.in` omits `tests/fixtures/plans_0_6_0/*.json` and `CHANGELOG.md`; running `pytest` from an unpacked sdist fails `test_saved_plan_compat.py`. |
| **H-08** | **High** | Spec & Docs | `IMPLEMENTATION.md:1, 28, 36`, `SPEC.md:3, 578, 914` | `SPEC.md` and `IMPLEMENTATION.md` still state version `0.1.2` and `Fifty-nine operators` (actual: `0.8.0` and `69` operators), omit `since` from `Arg`, and list shipped random initializers as future work. |
| **M-01** | **Medium** | Pretrained | `src/hndl/pretrained.py:102-169` | Process-global `@lru_cache` on `resolve_source` masks on-disk mutations to `config.json`/`.safetensors`/`.pth`, and `.safetensors` revision hashes only `config.json` + file size (`st_size`), missing in-place weight changes. |
| **M-02** | **Medium** | Pretrained | `src/hndl/pretrained.py:125, 134, 142, 338-347` | Missing `_require("safetensors")` in `TimmProvider.instantiate` (raises `ModuleNotFoundError` instead of `E_PRETRAINED`) and unwrapped `json.JSONDecodeError`/`OSError` in `resolve_source`. |
| **M-03** | **Medium** | Parser & Config | `src/hndl/_parser.py:286-287`, `src/hndl/config.py:117-118` | Empty tuple assignment `p = ()` is silently accepted in configs, and `"tensor_tuple"` AST records omit `"source"`, causing `E_OUTPUT_ARITY` to lack line/column info. |
| **M-04** | **Medium** | Parser | `src/hndl/_parser.py:100-141` | On Python 3.11, `tokenize.generate_tokens` emits `f"{...}"` as a single `STRING` token, allowing deeply nested expressions inside `{...}` to bypass `_screen` before `ast.parse`. |
| **M-05** | **Medium** | Capture & Types | `src/hndl/capture.py:75`, `src/hndl/types.py:169` | Falsy `input_dtype` values (`""`, `False`, `0`, `()`) silently fall back to the compute `dtype` via boolean `or` instead of raising `E_SCHEMA`. |
| **M-06** | **Medium** | Capture, Types & Resolver | `src/hndl/types.py:134`, `src/hndl/capture.py:78-80`, `src/hndl/resolver.py:462-501` | External input/output port names colliding with Python keywords (`for`, `in`), `range`, or `SCHEME_NAMES` (`normal`, `uniform`) are not rejected at contract definition; `resolve_graph` skips alias collision check. |
| **M-07** | **Medium** | Resolver | `src/hndl/resolver.py:438-441` | When a port has both a collapsed interval axis (e.g. `(8, 8)`) and an unresolved ambiguous axis (e.g. `(15, 16)`), `E_AMBIGUOUS` reports the collapsed `(8, 8)` interval. |
| **M-08** | **Medium** | Torch Backend | `src/hndl/torch.py:808-813` | `torch.compile` path in `_execute` skips pre-forward `self._verify_state("build")`, misreporting pre-forward state mutations as `"changed during forward"`. |
| **M-09** | **Medium** | Torch Backend | `src/hndl/torch.py:312-316, 324-339` | `copy.copy(model)` shares `_validated` and `nodes`; calling `.double()`/`.to()` on the shallow copy or on a child module (`model[0].double()`) mutates shared parameters while leaving `model` on the unvalidated fast path. |
| **M-10** | **Medium** | Spec / Plan Repr / Registry | `src/hndl/types.py:418-429`, `src/hndl/registry.py:169-186` | `ResolvedPlan.__repr__` always prints `"index"` and omits input port references on branched graphs (diverging from `SPEC.md:275, 320`); `positional_rest` replaces rather than follows `positional_names`. |
| **M-11** | **Medium** | Typing (`py.typed`) | `src/hndl/py.typed`, `src/hndl/*.py` | `py.typed` (PEP 561) is shipped, but public functions lack parameter/return type annotations and exported dataclasses type fields as `: object`. |
| **M-12** | **Medium** | Operators (`attention`, `pos_embed`, `moe`) | `src/hndl/operators/attention.py:69-77`, `pos_embed.py:66-67`, `moe.py:21-38, 125-132` | `_rope_tables` uses `float32` when cast to `float64`; `PosEmbed.forward` lacks `T > max_len` check (silently broadcasts when `max_len == 1`); `MixtureOfExperts` `_reference` duplicates sparse dispatch instead of dense sum and `nonzero`/`index_add` breaks `torch.compile` and CUDA deterministic mode. |
| **M-13** | **Medium** | Operator References & Tests | `src/hndl/operators/*.py`, `tests/operators/` | 20 of 69 operators omit `reference=` (skipped by `test_reference_implementation_matches`); 15 of 69 lack `tests/operators/test_<alias>.py`; 9 operators have neither `reference=` nor a dedicated test file. |
| **M-14** | **Medium** | CI/CD, Benchmarks & Docs | `.github/workflows/publish.yml:68-73`, `ci.yml:3-7`, `tests/benchmark/`, `scripts/gen_doc_pages.py` | `publish.yml` `cd "$RUNNER_TEMP"` does not persist to the `pytest` step; `ci.yml` does not run on `push` to `develop`; benchmark `device` fixture is unused (`DEVICE = "cpu"` hardcoded) and `HandwrittenTransformerBlock` uses fused `nn.MultiheadAttention` vs 3 `nn.Linear`s; `README.md` `https://` links bypass MkDocs link rewriting. |
| **L-01** | **Low** | Config Interpreter | `src/hndl/config.py:97-103` | Loop-suffixed node ID collision error message claims `(an explicit name='n0')` even when `'n0'` was auto-generated. |
| **L-02** | **Low** | Capture | `src/hndl/capture.py:127-189, 282-305` | `resolve_callable` passes `source=None` to `capture.emit`, so emit-time errors lack `node=` context even when `name="..."` was supplied (`SPEC.md:862`). |
| **L-03** | **Low** | Resolver | `src/hndl/resolver.py:374-387` | `_Solver.patterns` blames the first port with `ELLIPSIS` when `ellipsis_length < 0`, even if that port's shape was `None` and a later port had the invalid rank. |
| **L-04** | **Low** | Resolver & Relations | `src/hndl/resolver.py:284-291, 327-341`, `src/hndl/relations.py:59-72, 82-95` | `_Solver.axis`/`interval` do not bounds-check axis indices; `conv_transpose_input` can return `<= 0` instead of `None`; `conv_input_range(extent <= 0)` can return a non-empty interval. |
| **L-05** | **Low** | Types & Operator | `src/hndl/operator.py:284-293`, `src/hndl/types.py:257-264`, `src/hndl/resolver.py:449-458` | `Example.__post_init__` turns dict contracts into key tuples via `tuple(...)`; `Node`/`ResolvedNode` lack str/Mapping guards; `argument_origins` is duplicated in `ResolvedNode.source` and `ResolvedNode.provenance`. |
| **L-06** | **Low** | Torch Backend | `src/hndl/torch.py:995-1013` | `_state_bytes` meta-device estimation keys by `id(value)` rather than storage identity and assumes all floating tensors share `compute_dtype`. |
| **L-07** | **Low** | Operators (`chunk`, `grid_sample`) | `src/hndl/operators/chunk.py:141-144`, `grid_sample.py:12-16` | `Chunk.__init__` computes unused `self.size` attribute; `GridSample` leaves optional `mode`, `padding_mode`, and `align_corners` as `positional=True`. |
| **L-08** | **Low** | Operators (`extra_repr`, `resblock`, `transformer_block`) | `src/hndl/operators/coordinate_grid.py`, `fourier_features.py`, `grid_sample.py`, `_equalized.py`, `attention.py`, `resblock.py`, `transformer_block.py` | Missing `extra_repr()` on 3 operators; `equalized` omitted from `EqualLinear`/`Attention.extra_repr()`; `ResBlock.__init__` lacks direct-call `_validate`; `transformer_block._reference` duplicates attention/FFN reference math. |
| **L-09** | **Low** | Docs & Benchmark Comments | `docs/ADDING_OPERATORS.md:217`, `tests/benchmark/_timing.py:10`, `test_parity_vs_handwritten_pytorch.py:41` | `ADDING_OPERATORS.md` omits `scripts` from `ruff check`; benchmark comments reference non-existent `"torch 2.14"`. |
| **L-10** | **Low** | Repo Hygiene | `.claude/worktrees/`, `pyproject.toml:48-53` | 28 prunable git worktrees, 2 nested worktrees in `.claude/worktrees/`, and ~90 stale local automation branches; `pyproject.toml` lacks `norecursedirs`. |

---

## 2. Subsystem Architecture & Inventory Review

### 2.1 Core Engine Modules (`src/hndl/*.py`)

| Module | Lines | Assessment & Key Observations |
| :--- | ---: | :--- |
| `src/hndl/__init__.py` | 17 | Public package root exporting `__version__`, `Arg`, `Example`, `HNDLError`, `Policy`, `Registry`, `ResolvedPlan`, `operator`, `ops`, `resolve`, `resolve_callable`, `resolve_file`, and the 8 initializer schemes. Does not import `torch`. |
| `src/hndl/_version.py` | 4 | Single version string `"0.8.0"`, matching `pyproject.toml`. |
| `src/hndl/errors.py` | 37 | Defines `HNDLError(ValueError)` with structured `code`, `message`, `node`, `line`, `column`, and loop `iterations`, plus `describe_iterations` and `source_location`. |
| `src/hndl/_parser.py` | 390 | Token screener (`_screen`) and AST allowlist validator (`Validator`). Bounds source at 64 KiB, bracket depth at 50, loop depth at 8, and operator/keyword count at 32. See **H-01**, **H-02**, **M-03**, **M-04**. |
| `src/hndl/config.py` | 184 | Evaluates statement records (`_Interpreter`) into a `Capture`, unrolling `for _ in range(N):` loops and suffixing explicit node names (`block0`, `block1_3`). See **M-03**, **L-01**. |
| `src/hndl/capture.py` | 333 | Context-scoped graph builder (`Capture`), `Symbol` edge references (blocking eager Python/PyTorch operations), and `OperatorNamespace` (`ops` and `registry.ops`). See **M-05**, **M-06**, **L-02**. |
| `src/hndl/operator.py` | 712 | `@operator` decorator, `Arg`, `Sym`, `Port`, `Policy`, `Example`, `Operator`, `parse_shape`, and `NodeView` solver facade. See **M-10**, **L-05**. |
| `src/hndl/registry.py` | 211 | `Registry` (holding operators, `pretrained_provider` builders, and `pretrained_readout` callables) and `normalize_arguments`. See **M-10**. |
| `src/hndl/relations.py` | 256 | Public shape arithmetic (`conv_output`, `conv_input_range`, `conv_transpose_output`, `conv_transpose_input`) and relation helpers (`conv_axis`, `conv_transpose_axis`, `spatial`, `elementwise_join`, `broadcast`). See **L-04**. |
| `src/hndl/resolver.py` | 622 | DAG validation, dead-node/unused-input detection, static edge dtype checking, `_Solver` bidirectional fixed-point propagation, and `validate_concrete_plan` restoration verifier. See **M-06**, **M-07**, **L-03**, **L-04**, **L-05**. |
| `src/hndl/settings.py` | 216 | Normalization and validation of `init=` (`torch_default@1` constants and `torch_default@2` random schemes) and `trainable=` metadata, with canonical `float32` struct rounding. |
| `src/hndl/initializers.py` | 58 | Pure data builders for the 8 `torch.nn.init` schemes (`xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `truncated_normal`, `normal`, `uniform`, `orthogonal`). |
| `src/hndl/types.py` | 437 | Immutable `Node`, `Graph`, `ResolvedNode`, `ResolvedPlan` dataclasses, batch-multiple helpers (`"B"`, `"k*B"`), canonical JSON serialization, and SHA-256 digests. See **M-05**, **M-06**, **M-10**, **L-05**. |
| `src/hndl/torch.py` | 1,262 | `GraphModule`, `Network`, `_NodeModules`, signature-cached validation, slot-indexed `_fast` execution loop, C-level store mutation detection, `deepcopy`, meta-device `max_state_bytes` budgeting, and RNG-scoped initialization. See **H-03**, **H-04**, **M-08**, **M-09**, **L-06**. |
| `src/hndl/pretrained.py` | 593 | `Source` resolution, `TransformersProvider`, `TimmProvider`, `LocalProvider` (`.pth` with `sha256`, `layer=`, `layers=`, `readout=`), and meta-device output shape tracing. See **H-05**, **M-01**, **M-02**. |
| `src/hndl/testing.py` | 389 | Public operator verification harness (`check_operator`, `check_declaration`, `check_round_trip`, `check_build_and_run`, `check_reference`). |
| `src/hndl/docs.py` | 212 | Generates `docs/operators/*.md`, `docs/operators/index.md`, and `docs/networks.md` with `--check` mode for CI. |

### 2.2 Complete Operator Catalog (`src/hndl/operators/*.py`)

All **69 built-in operators** across **12 categories** (plus 4 private support modules: `__init__.py`, `_equalized.py`, `_pooling.py`, `_relations.py`):

| Category | Count | Operators (`alias`) |
| :--- | ---: | :--- |
| **`activation`** | 13 | `elu`, `gelu`, `hardswish`, `identity`, `leaky_relu`, `mish`, `quick_gelu`, `relu`, `sigmoid`, `silu`, `softmax`, `softplus`, `tanh` |
| **`arithmetic`** | 9 | `add`, `broadcast_add`, `broadcast_mul`, `clamp`, `constant`, `learned_scale`, `mul`, `scale`, `sub` |
| **`convolution`** | 3 | `conv` (`conv2d@1`), `conv1d` (`conv1d@1`), `deconv` (`conv_transpose2d@1`) |
| **`core`** | 1 | `linear` |
| **`memory`** | 1 | `hopfield` |
| **`normalization`** | 6 | `adaptive_norm`, `batch_norm`, `group_norm`, `instance_norm`, `layer_norm`, `rms_norm` |
| **`pretrained`** | 1 | `pretrained` |
| **`regularization`** | 1 | `dropout` |
| **`sequence`** | 12 | `attention`, `cls_token`, `cross_attention`, `embedding`, `feed_forward`, `fourier_features`, `gather_token`, `moe`, `pool_tokens`, `pos_embed`, `swiglu`, `transformer_block` |
| **`shape`** | 10 | `chunk`, `concat`, `flatten`, `mean`, `pad`, `permute`, `reshape`, `split`, `sum`, `transpose` |
| **`spatial`** | 9 | `adaptive_avg_pool` (`adaptive_avg_pool2d@1`), `avg_pool` (`avg_pool2d@1`), `coordinate_grid`, `global_avg_pool`, `grid_sample`, `max_pool` (`max_pool2d@1`), `pixel_shuffle`, `spatial_attention`, `upsample` |
| **`vision`** | 2 | `patch_embed`, `resblock` |

---

## 3. Detailed Findings & Recommendations

### 3.1 High-Severity Findings (`H-01` – `H-08`)

#### [H-01] Unhandled `IndexError` on `\r` (Carriage Return) Line Breaks in `parse()`
- **Component:** Declarative Parser (`src/hndl/_parser.py:156, 364-366`)
- **Symbols:** `parse`, `Validator.location`
- **Evidence:**
  ```python
  # src/hndl/_parser.py:364-366
  original_lines = source.split("\n")
  clean_lines = dedented.split("\n")
  offsets = [max(0, len(original) - len(clean)) for original, clean in zip(original_lines, clean_lines)]
  ```
  ```python
  # src/hndl/_parser.py:153-157
  def location(self, node):
      prefix = self.lines[node.lineno - 1].encode("utf-8")[:node.col_offset].decode("utf-8", "ignore")
      return {"line": node.lineno, "column": len(prefix) + 1 + self.offsets[node.lineno - 1]}
  ```
- **Root Cause & Impact:** Python's `tokenize.generate_tokens` and `ast.parse` treat bare `\r` as a line break and increment `lineno` on each `\r`. Because `parse()` splits `original_lines` and `clean_lines` strictly on `"\n"`, a config string containing `\r` produces AST nodes with `node.lineno > len(self.lines)`, raising an uncaught `IndexError: list index out of range` instead of an `HNDLError` (violating `SPEC.md` §13's closed error contract).
- **Verified Reproducer:**
  ```python
  import hndl
  hndl.resolve("linear(64)\rrelu()\rlinear()", input_shape=("B", 128), output_shape=("B", 10))
  # Raises IndexError: list index out of range at src/hndl/_parser.py:156
  ```
- **Recommendation:** Normalize `\r\n` and `\r` to `\n` in `_source_text` (or split lines with `.splitlines()`), and guard `self.lines` / `self.offsets` indexing with bounds checks.

---

#### [H-02] Token Screener (`_screen`) False Positive Across Newlines for Tuple/List Unpacking
- **Component:** Declarative Parser (`src/hndl/_parser.py:60, 86-91, 136-137`)
- **Symbols:** `_LAYOUT`, `_counts`, `_screen`
- **Evidence:**
  ```python
  # src/hndl/_parser.py:60
  _LAYOUT = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)
  ```
  ```python
  # src/hndl/_parser.py:86-91
  if token.string in _OPENERS:
      return previous is not None and (previous.type in (tokenize.NUMBER, tokenize.STRING)
                                       or previous.string in _CLOSERS)
  ```
  ```python
  # src/hndl/_parser.py:136-137
  if token.type not in _LAYOUT:
      previous = token
  ```
- **Root Cause & Impact:** Because `tokenize.NEWLINE` is in `_LAYOUT`, `_screen` does not reset `previous` at the end of a logical line. Every normal HNDL statement ends with `)` (`in _CLOSERS`). When the following statement starts with `(` or `[` for tuple/list unpacking (e.g., `(a, b) = split(x, 32)` or `[a, b] = split(x, 32)`, which `_parser.py:183` explicitly supports), `_counts` sees `previous.string == ")"` followed by `(` or `[` and counts it as a chained call or subscript (`relu()()`). Once a config has 33 such unpacking statements (e.g., across a multi-stage network), `_screen` aborts with `E_SYNTAX: Source uses more than 32 Python operators or keywords`.
- **Verified Reproducer:**
  ```python
  import hndl
  cfg = "linear(64)\n" + "(a, b) = split(x, 32)\n" * 33 + "add(a, b)\n"
  hndl.resolve(cfg, input_shape=("B", 64), output_shape=("B", 32))
  # Raises HNDLError: E_SYNTAX (line 34, column 1): Source uses more than 32 Python operators or keywords
  ```
- **Recommendation:** Remove `tokenize.NEWLINE` from `_LAYOUT` and reset `previous = None` when `token.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)` (leaving `tokenize.NL` in `_LAYOUT` so multiline parenthesized expressions preserve `previous`).

---

#### [H-03] `_is_chain` False Positive on Single-Output Tuple-Returning Nodes (`chunk(1)`, `pretrained(..., layers=("...",))`) Breaks Slicing
- **Component:** PyTorch Backend (`src/hndl/torch.py:178-189, 883-885`)
- **Symbols:** `_is_chain`, `GraphModule.__getitem__`
- **Evidence:**
  ```python
  # src/hndl/torch.py:178-189
  def _is_chain(plan):
      """A chain has one external input feeding one unary node after another."""
      if len(plan.inputs) != 1 or len(plan.outputs) != 1:
          return False
      previous = f"input:{next(iter(plan.inputs))}"
      for node in plan.nodes:
          if (len(node.inputs) != 1 or tuple(node.inputs.values()) != (previous,)
                  or len(node.outputs) != 1):
              return False
          previous = f"node:{node.id}/{node.outputs[0]}"
      return plan.output_ref == previous
  ```
- **Root Cause & Impact:**
  1. Operators with variadic output ports (`chunk(1)` or `pretrained(..., layers=("layer1",))`) have `len(node.outputs) == 1` (`outputs == ("out0",)`), clear `current`, and return a 1-tuple `(tensor,)` from `forward()`.
  2. Because `_is_chain(plan)` checks `len(node.outputs) != 1` instead of `node.outputs != ("out",)` (or `spec.returns_tuple(node.args)`), a graph like `a, = chunk(1); linear(a, 10)` is misclassified as `_chain = True`.
  3. Consequently, `model[:2]` returns an `nn.Sequential` chaining `Chunk(1)` into `Linear`, which crashes on forward with `TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple`, and `repr(model)` prints the chain table format instead of the port-mapped graph table.
  4. Additionally, `GraphModule.__getitem__` (`src/hndl/torch.py:883-885`) allows slices with `key.step not in (None, 1)` (such as `model[::-1]` or `model[::2]`), producing an invalid `nn.Sequential` whose layers are non-adjacent or reversed.
- **Verified Reproducer:**
  ```python
  import torch
  from hndl.torch import network
  model = network("a, = chunk(1)\nlinear(a, 10)", input_shape=("B", 8), output_shape=("B", 10), device="cpu")
  model[:2](torch.randn(2, 8))
  # Raises TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
  ```
- **Recommendation:**
  - In `_is_chain(plan)`, require `tuple(node.outputs) == ("out",)` (so variadic-output nodes with port `"out0"` are treated as branched).
  - In `GraphModule.__getitem__`, reject `key.step not in (None, 1)` with `ValueError` or `TypeError`.

---

#### [H-04] `pretrained(..., trainable=True)` Leaves the Wrapped Model in `eval()` Mode After Construction
- **Component:** PyTorch Backend & `pretrained` Operator (`src/hndl/operators/pretrained.py:208-216`, `src/hndl/torch.py:1201-1216`)
- **Symbols:** `Pretrained.__init__`, `Pretrained.train`, `_build`
- **Evidence:**
  ```python
  # src/hndl/operators/pretrained.py:207-216
  self.model = self.provider.instantiate(weights=materialize)
  self.model.eval()
  for parameter in self.model.parameters():
      parameter.requires_grad_(False)

  def train(self, mode=True):
      super().train(mode)
      if not any(parameter.requires_grad for parameter in self.model.parameters()):
          self.model.eval()
      return self
  ```
  ```python
  # src/hndl/torch.py:1201, 1210-1216
  layer = construct(specs[node.id], node, device, dtype)
  ...
  with torch.no_grad():
      for node, assignments in zip(plan.nodes, settings):
          for parameter, target, override, trainable in assignments:
              if override is not None:
                  _apply_initializer(parameter, override, target, node.id)
              if trainable is not None:
                  parameter.requires_grad_(trainable)
  ```
- **Root Cause & Impact:** `README.md:216`, `docs/pretrained.md:223-227`, and `src/hndl/operators/pretrained.py:116-117` state that the wrapped model stays frozen and in eval mode *unless* `trainable=True` is passed. However, `Pretrained.__init__` unconditionally calls `self.model.eval()` and sets `requires_grad_(False)`. Later in `_build`, `parameter.requires_grad_(True)` is applied to the parameters, but `layer.train(layer.training)` is never called afterward. Thus, a freshly constructed `network('pretrained(..., trainable=True)', ...)` has `model.training == True` and `model[0].training == True`, but `model[0].model.training == False`! Unless the caller manually calls `model.train()` after `network(...)`, dropout and batchnorm inside the trainable pretrained model remain stuck in `eval` mode during training.
- **Recommendation:** In `_build` (`src/hndl/torch.py:1216`), after applying `parameter.requires_grad_(trainable)` on a node's module, call `modules[f"n_{node.id}"].train(modules[f"n_{node.id}"].training)` so submodules whose `train()` behavior depends on `requires_grad` (such as `Pretrained`) synchronize their training mode.

---

#### [H-05] `TimmProvider.instantiate` Drops `self.source.revision` for `hf://owner/repo@revision` Sources
- **Component:** Pretrained Loader (`src/hndl/pretrained.py:114-126, 344-345`)
- **Symbols:** `resolve_source`, `TimmProvider.instantiate`
- **Evidence:**
  ```python
  # src/hndl/pretrained.py:114, 126
  repo, _, revision = source[len(_HF_PREFIX):].partition("@")
  ...
  return Source("hf", repo, commit, config, str(snapshot), _provider_for(config))
  ```
  ```python
  # src/hndl/pretrained.py:344-345
  if self.source.kind == "hf":
      return self.timm.create_model(f"hf-hub:{self.source.location}", pretrained=True, **kwargs)
  ```
- **Root Cause & Impact:** When `source` is `"hf://timm/resnet18.a1_in1k@<revision>"`, `resolve_source` strips `@<revision>` from `self.source.location` (`"timm/resnet18.a1_in1k"`) and records the resolved commit/revision in `self.source.revision`. While `TransformersProvider.instantiate` (`pretrained.py:281`) passes `revision=self.source.revision` to `from_pretrained`, `TimmProvider.instantiate` passes only `f"hf-hub:{self.source.location}"` without the revision. Consequently, `timm` downloads weights from the default branch rather than the revision recorded in the resolved plan.
- **Recommendation:** Pass `f"hf-hub:{self.source.location}@{self.source.revision}"` (or download `model.safetensors` at `revision=self.source.revision` via `huggingface_hub.hf_hub_download` and load via `pretrained_cfg_overlay={"file": ...}`).

---

#### [H-06] `RMSNorm.forward` Unconditionally Casts `float64` Inputs to `float32`
- **Component:** Operators (`src/hndl/operators/rms_norm.py:76-80`)
- **Symbols:** `RMSNorm.forward`
- **Evidence:**
  ```python
  # src/hndl/operators/rms_norm.py:76-80
  def forward(self, x):
      statistic = x.float()
      scale = torch.rsqrt(statistic.square().mean(dim=-1, keepdim=True) + self.eps)
      normalized = (statistic * scale).to(x.dtype)
      return normalized * self.weight if self.weight is not None else normalized
  ```
- **Root Cause & Impact:** `GraphModule._apply` (`src/hndl/torch.py:324-350`) and `README.md:76` explicitly support casting built networks to `float64` via `.double()`. However, `RMSNorm.forward` calls `x.float()` unconditionally instead of only upcasting reduced-precision inputs (`float16`, `bfloat16`). When called with `float64` inputs, `x.float()` truncates 53-bit double precision to 24-bit single precision before computing the RMS normalization, causing ~`3e-8` error against `torch.nn.functional.rms_norm` (`_reference`) and breaking `float64` numerical gradchecks.
- **Recommendation:** Change line 77 to:
  ```python
  statistic = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
  ```

---

#### [H-07] `MANIFEST.in` Omits `tests/fixtures/plans_0_6_0/*.json` and `CHANGELOG.md`, Breaking `pytest` from sdist
- **Component:** Packaging (`MANIFEST.in:1-4`, `tests/test_saved_plan_compat.py:21-29`)
- **Evidence:**
  ```text
  # MANIFEST.in
  include LICENSE README.md SPEC.md IMPLEMENTATION.md CONTRIBUTING.md
  recursive-include tests *.py
  recursive-include examples *.py *.hndl *.json *.md
  recursive-include docs *.md
  ```
- **Root Cause & Impact:**
  1. `recursive-include tests *.py` excludes the 5 JSON fixture files in `tests/fixtures/plans_0_6_0/*.json` (`attention.json`, `feed_forward.json`, `gpt_tiny.json`, `linear_mlp.json`, `linear_options.json`).
  2. Running `pytest` from an unpacked source distribution (`sdist`) fails in `tests/test_saved_plan_compat.py:25-29` (`test_fixtures_cover_the_operators_that_gained_arguments`) because `FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))` is empty!
  3. `CHANGELOG.md` is referenced in `README.md:500` and staged into MkDocs by `scripts/gen_doc_pages.py:29`, but is omitted from `MANIFEST.in`.
- **Recommendation:** Update `MANIFEST.in` to:
  ```text
  include LICENSE README.md CHANGELOG.md SPEC.md IMPLEMENTATION.md CONTRIBUTING.md
  recursive-include tests *.py *.json
  recursive-include examples *.py *.hndl *.json *.md
  recursive-include docs *.md
  ```

---

#### [H-08] Stale Release Version (`0.1.2`), Operator Count (`59`), and Outdated Spec Sections in `IMPLEMENTATION.md` and `SPEC.md`
- **Component:** Documentation & Specification (`IMPLEMENTATION.md:1, 28, 36`, `SPEC.md:3, 578, 914`)
- **Evidence:**
  - `IMPLEMENTATION.md:1`: `# Implementation notes: 0.1.2`
  - `IMPLEMENTATION.md:28`: `- Fifty-nine operators across activation, arithmetic, shape, normalization...` (there are **69** operators across **12** categories, including `core`).
  - `IMPLEMENTATION.md:36`: `- Persisted constant parameter initialization and trainability overrides...` (omits the 8 random initializer schemes described at lines 212–250).
  - `SPEC.md:3`: `**Status: the implemented 0.1.2 contract.** ... [IMPLEMENTATION.md](IMPLEMENTATION.md) records what 0.1.2 ships and what remains for v1.`
  - `SPEC.md:578`: `Arg(type, default, *, min, max, exclusive_min, exclusive_max, help, inferable, dim, positional, choices)` (omits `since`, added in `0.7.0`).
  - `SPEC.md:914`: `- Extend initialization beyond constant overrides once seeding, device-placement, and reproducibility semantics are specified.` (shipped in `0.6.0` as `torch_default@2`).
- **Recommendation:** Update `SPEC.md` and `IMPLEMENTATION.md` headers to `0.8.0` (and add a test in `tests/test_package.py` similar to `test_readme_status_names_the_current_version` so they never drift on future releases), update the operator count to 69, add `since` to `SPEC.md:578`, and remove the obsolete bullet at `SPEC.md:914`.

---

### 3.2 Medium-Severity Findings (`M-01` – `M-14`)

#### [M-01] Process-Global `@lru_cache` on `resolve_source` Masks On-Disk Checkpoint Edits & `.safetensors` Revision Ignores Weight Content
- **Component:** Pretrained Loader (`src/hndl/pretrained.py:102-169`)
- **Symbols:** `resolve_source`, `file_sha256`
- **Evidence:**
  ```python
  # src/hndl/pretrained.py:102-103
  @lru_cache(maxsize=64)
  def resolve_source(source, config_path="", provider="", checksum=""):
  ```
  ```python
  # src/hndl/pretrained.py:143
  revision = _digest(config_file.read_bytes(), str(path.stat().st_size))
  ```
- **Root Cause & Impact:**
  1. While `file_sha256` (`pretrained.py:72`) keys its cache by `(str(path), stat.st_size, stat.st_mtime_ns)`, `resolve_source` is decorated with `@lru_cache(maxsize=64)` keyed only by its string arguments `(source, config_path, provider, checksum)`. If a local `config.json`, `.safetensors`, or `.pth` file changes on disk within the same Python process (e.g. between `resolve()` and `ResolvedPlan.from_json()`), `resolve_source` returns the cached `Source` without re-checking disk state.
  2. For a local `.safetensors` file (`kind == "file"`, line 143) or directory (`kind == "directory"`, line 135), `revision` hashes only `config.json` (and `st_size` for `file`), ignoring `st_mtime_ns` and weight bytes. Replacing weights with same-shape tensors of identical byte length does not change `revision`.
- **Recommendation:** Remove `@lru_cache` from `resolve_source` for local paths (or include `(stat.st_size, stat.st_mtime_ns)` of both `config.json` and weight files in the cache key and local revision digest).

---

#### [M-02] Missing `_require("safetensors")` in `TimmProvider.instantiate` and Unwrapped IO/JSON Exceptions in `resolve_source`
- **Component:** Pretrained Loader (`src/hndl/pretrained.py:125, 134, 142, 338-341`)
- **Evidence:**
  ```python
  # src/hndl/pretrained.py:337-339
  if weights:
      from safetensors.torch import load_file
      missing, unexpected = model.load_state_dict(load_file(self.source.location), strict=False)
  ```
- **Root Cause & Impact:**
  1. Unlike `TransformersProvider._load_safetensors` (`pretrained.py:288`), `TimmProvider.instantiate` imports `safetensors.torch` without calling `_require("safetensors")`. If `safetensors` is missing, it raises `ModuleNotFoundError` instead of `HNDLError("E_PRETRAINED", ...)`.
  2. In `resolve_source` (`lines 125, 134, 142`), `json.loads(config_file.read_text(encoding="utf-8"))` is not wrapped in a `try...except`, so invalid UTF-8 or malformed JSON in `config.json` raises `UnicodeDecodeError` or `json.JSONDecodeError` instead of `HNDLError("E_PRETRAINED", ...)`.
- **Recommendation:** Call `_require("safetensors")` in `TimmProvider.instantiate` and wrap `config.json` reading/parsing in `try ... except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc: _fail(...)`.

---

#### [M-03] Empty Tuple Assignment `p = ()` Silently Accepted & Missing Source Location on `tensor_tuple` Arity Error
- **Component:** Declarative Parser & Config Interpreter (`src/hndl/_parser.py:286-287`, `src/hndl/config.py:117-118`)
- **Evidence:**
  ```python
  # src/hndl/_parser.py:286-287
  if isinstance(node, ast.Tuple) and all(isinstance(item, (ast.Name, ast.Call)) for item in node.elts):
      return {"kind": "tensor_tuple", "items": [self.expression(item, assignment=True) for item in node.elts]}
  ```
  ```python
  # src/hndl/config.py:116-118
  values = [self.evaluate(item) for item in expression["items"]]
  if kind == "tensor_tuple" and not all(isinstance(item, Symbol) for item in values):
      raise HNDLError("E_OUTPUT_ARITY", "A tensor tuple must contain only tensor symbols")
  ```
- **Root Cause & Impact:**
  1. When `node` is an empty tuple `()`, `node.elts` is empty, so `all(...)` evaluates to `True` in both `_parser.py:286` and `config.py:117`, allowing `p = ()` in declarative configs without raising `E_SYNTAX`.
  2. The `"tensor_tuple"` record omits `"source": self.location(node)`, and `config.py:118` raises `HNDLError` without `self.where(expression["source"])`, omitting line and column numbers when a multi-output call is placed inside a tensor tuple (`a = split(4); b = (a, x)`).
- **Recommendation:** Require `node.elts` to be non-empty at `_parser.py:286`, attach `"source": self.location(node)` to `"tensor_tuple"`, and pass `self.where(expression["source"])` to `_error(...)` at `config.py:118`.

---

#### [M-04] Python 3.11 f-string Bypass of `_screen` Token Screener
- **Component:** Declarative Parser (`src/hndl/_parser.py:100-141`)
- **Root Cause & Impact:** On Python 3.12+ (PEP 701), `tokenize.generate_tokens` tokenizes the interior of `f"{...}"` expressions so `_screen` bounds their bracket depth and operator count. On Python 3.11 (which is supported in `pyproject.toml` and CI), `tokenize.generate_tokens` emits the entire `f"{...}"` literal as a single `tokenize.STRING` token. `ast.parse` then parses the embedded expression in C without `_screen` screening its nesting depth.
- **Recommendation:** In `_screen`, reject any `tokenize.STRING` token carrying an `f` or `F` prefix with `E_SYNTAX` before `ast.parse` is called.

---

#### [M-05] Falsy `input_dtype` Values (`""`, `False`, `0`, `()`) Silently Fall Back to Compute `dtype`
- **Component:** Symbolic Capture & Types (`src/hndl/capture.py:75`, `src/hndl/types.py:169`)
- **Evidence:**
  ```python
  # src/hndl/capture.py:75-76
  self.input_contracts = {name: {"shape": shape, "dtype": dtypes[name] or dtype}
                          for name, shape in shapes.items()}
  ```
  ```python
  # src/hndl/types.py:168-169
  inputs[name] = {"shape": _shape(entry["shape"], f"input_shape[{name}]"),
                  "dtype": entry.get("dtype") or record.dtype}
  ```
- **Root Cause & Impact:** Using `dtypes[name] or dtype` treats any falsy non-`None` value (`""`, `False`, `0`, `[]`, `()`) as omitted and silently substitutes `dtype` (`"float32"`) instead of letting `resolve_graph` reject it with `E_SCHEMA`.
- **Recommendation:** Use `dtype if dtypes[name] is None else dtypes[name]` in `src/hndl/capture.py:75` and `record.dtype if entry.get("dtype") is None else entry.get("dtype")` in `src/hndl/types.py:169`.

---

#### [M-06] External Port Names Colliding with Python Keywords, `range`, or `SCHEME_NAMES` Not Rejected at Contract Definition
- **Component:** Types, Capture & Resolver (`src/hndl/types.py:134`, `src/hndl/capture.py:78-80`, `src/hndl/resolver.py:462-501`)
- **Root Cause & Impact:**
  1. `named_contracts` (`src/hndl/types.py:134`) checks only `_PORT_NAME.fullmatch(name)` (`[a-z][a-z0-9_]*`), and `Capture.__init__` (`src/hndl/capture.py:78`) checks only collisions with `self.registry.aliases`.
  2. External input/output port names that are Python keywords (`"for"`, `"in"`, `"if"`, `"def"`), `"range"`, or initializer scheme names (`"normal"`, `"uniform"`) pass contract validation in `Capture`, even though they cannot be bound as keyword parameters in `resolve_callable` or written as identifiers in declarative configs.
  3. Direct calls to `resolve_graph(graph, registry=...)` (`src/hndl/resolver.py:462-501`) do not check `(set(graph.inputs) | set(graph.outputs)) & set(registry.aliases)`.
- **Recommendation:** Reject `keyword.iskeyword(name)` in `named_contracts` (just as `_validate_name` in `src/hndl/operator.py:296` does for operator ports) and reject `LOOP_ITERATOR` (`"range"`), `SCHEME_NAMES`, and `registry.aliases` collisions in both `Capture.__init__` and `resolve_graph`.

---

#### [M-07] Misleading `E_AMBIGUOUS` Interval Reported When Another Axis on the Same Port Collapsed
- **Component:** Constraint Solver (`src/hndl/resolver.py:438-441`)
- **Evidence:**
  ```python
  # src/hndl/resolver.py:438-441
  if shape is None or any(value is None for value in shape):
      interval = next((bounds for (r, _), bounds in self.intervals.items() if r == ref), None)
      detail = f"; inverse allows {interval}" if interval else ""
      self.error("E_AMBIGUOUS", f"Shape for {port} ({ref}) remains ambiguous{detail}; supply a dimension or a policy")
  ```
- **Root Cause & Impact:** When a 2D convolution has `stride=(1, 2)` and its output feeds a layer that fixes the output shape, axis 2 (stride 1) records a collapsed interval `self.intervals[ref, 2] = (8, 8)` and resolves `shape[2] = 8`, while axis 3 (stride 2) records `self.intervals[ref, 3] = (7, 8)` and leaves `shape[3] = None`. Because line 439 looks up the first interval matching `r == ref` without checking `bounds[0] < bounds[1]` (or `shape[axis] is None`), the error message reports `; inverse allows (8, 8)` instead of `; inverse allows (7, 8)`.
- **Verified Reproducer:**
  ```python
  import hndl
  hndl.resolve(
      "conv(16, kernel_size=3, stride=(1, 2), padding=1)\navg_pool(2)",
      input_shape=("B", 3, 8, 16),
      output_shape=("B", 16, 4, 4),
  )
  # HNDLError: E_AMBIGUOUS (node n0; line 1, column 1): Shape for out (node:n0/out) remains ambiguous; inverse allows (8, 8); supply a dimension or a policy
  ```
- **Recommendation:** Change line 439 to select only non-collapsed intervals on unresolved axes:
  ```python
  interval = next((bounds for (r, ax), bounds in self.intervals.items()
                   if r == ref and bounds[0] < bounds[1]), None)
  ```

---

#### [M-08] `torch.compile` Path in `GraphModule._execute` Skips Pre-Forward `_verify_state("build")`
- **Component:** PyTorch Backend (`src/hndl/torch.py:808-813`)
- **Evidence:**
  ```python
  # src/hndl/torch.py:808-813
  if _is_compiling():
      outputs = self._run_checked(inputs, compiling=True)
      self._verify_state("forward")
      return outputs
  ```
- **Root Cause & Impact:** Unlike `_validate` (`src/hndl/torch.py:794`), which calls `self._verify_state("build")` *before* `_run_checked` and `self._verify_state("forward")` *after*, the `_is_compiling()` path only calls `self._verify_state("forward")` after `_run_checked`. If registered state was mutated before the compiled call, it either crashes inside TorchDynamo during `module(*bound)` or is misreported as `"changed during forward"` instead of `"no longer matches the build"`.
- **Recommendation:** Add `self._verify_state("build")` before `outputs = self._run_checked(inputs, compiling=True)` at `src/hndl/torch.py:811`.

---

#### [M-09] Shallow Copy (`copy.copy(model)`) and Child Module `.to()` / `.double()` Desynchronize Fast-Path Cache
- **Component:** PyTorch Backend (`src/hndl/torch.py:312-316, 324-339`)
- **Evidence:**
  ```python
  # src/hndl/torch.py:312-316
  def __copy__(self):
      """A second handle on the same graph: every tensor stays shared."""
      result = type(self).__new__(type(self))
      result.__dict__.update(self.__dict__)
      return result
  ```
  ```python
  # src/hndl/torch.py:337-338
  self._validated = {}
  self._state_seen = None
  ```
- **Root Cause & Impact:**
  1. `copy.copy(model)` creates a shallow copy sharing `self.nodes` and initially sharing `self._validated`. When `.double()` or `.to(...)` is called on the shallow copy, `_apply` mutates the shared parameters in-place and rebinds `copy_model._validated = {}` to a new dict without clearing the original `model._validated` dict or updating `model._runtime_dtype`. Subsequent calls to `model(x)` take the fast path with stale dtype expectations and fail inside PyTorch.
  2. Similarly, calling `model[0].double()` or `model[:1].double()` directly on a child module mutates `param.data` in-place without firing registration hooks or changing `_record_stores` (dictionary lengths and `None` slots stay identical), leaving `model._validated` populated so the next call bypasses `_check` and crashes inside PyTorch.
- **Recommendation:** Call `self._validated.clear()` before `self._validated = {}` in `_apply`, and either include parameter `(dtype, device)` in `_record_stores` or document that dtype/device casts must be performed on the root `GraphModule`/`Network`.

---

#### [M-10] `ResolvedPlan.__repr__` and `positional_rest` Diverge from `SPEC.md`
- **Component:** Plan Representation & Argument Normalization (`src/hndl/types.py:418-429`, `src/hndl/registry.py:169-186`, `src/hndl/operator.py:581-583`, `SPEC.md:275, 320, 534`)
- **Evidence:**
  1. `SPEC.md:275, 320` states that `print(plan)` / `repr(plan)` is the equivalent shape table to `print(model)`, where *"The index column is present only for a consecutive unary chain. A branched graph prints one row per node with every named port and its producing reference."* However, `ResolvedPlan.__repr__` (`src/hndl/types.py:422-426`) always prints the `"index"` column and omits `node.inputs[key]` (the producing reference) on branched graphs.
  2. `SPEC.md:534` and `docs/ADDING_OPERATORS.md:104` state that `positional_rest` receives *"leftover positional literals"*. In `src/hndl/registry.py:169-186` and `src/hndl/capture.py:181-183`, `if op.positional_rest is not None:` completely bypasses `op.positional_names`, routing *all* positional arguments into `positional_rest`.
- **Recommendation:** Align `ResolvedPlan.__repr__` with `GraphModule.__repr__` (or update `SPEC.md:320` to reflect the compact plan table), and in `make_operator` (`src/hndl/operator.py:581-583`) forbid setting `positional=True` on other `Arg`s when `positional_rest` is used (except the `positional_rest` arg itself).

---

#### [M-11] PEP 561 `py.typed` Marker Shipped While Public APIs Lack Type Annotations
- **Component:** Static Typing (`src/hndl/py.typed`, `pyproject.toml:46`, `src/hndl/*.py`)
- **Root Cause & Impact:** `src/hndl/py.typed` is included in the wheel (`pyproject.toml:46`), advertising PEP 561 inline type support to `mypy` and `pyright`. However, public functions (`resolve`, `resolve_file`, `resolve_callable`, `network`, `network_file`, `network_from_callable`, `build`, `parameter_counts`, `operator`, `check_operator`, etc.) have no parameter or return type annotations, and exported dataclasses (`Arg`, `Node`, `Graph`, `ResolvedNode`, `ResolvedPlan`) annotate fields as `: object` (`input_dtype: object = None`, `registry: object = None`, `source: object = None`), which causes strict type checkers to reject attribute access and indexing on those fields.
- **Recommendation:** Add type annotations to all public API entrypoints in `src/hndl/` and replace `: object` field annotations on public dataclasses with concrete types (`str | Mapping[str, str] | None`, `Registry | None`, etc.).

---

#### [M-12] Numerical & Runtime Edge Cases in `attention`, `pos_embed`, and `moe`
- **Component:** Built-in Operators (`src/hndl/operators/attention.py:69-77`, `src/hndl/operators/pos_embed.py:66-67`, `src/hndl/operators/moe.py:21-38, 125-132`)
- **Evidence & Issues:**
  1. **`attention._rope_tables` (`src/hndl/operators/attention.py:69-77`):** Computes rotary position frequencies and angles in `torch.float32` even when `dtype == torch.float64` (via `.double()`), losing double precision in the precomputed `rope_cos` and `rope_sin` buffers.
     - *Fix:* Use `calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32`.
  2. **`PosEmbed.forward` (`src/hndl/operators/pos_embed.py:66-67`):** Executes `return x + self.weight[:x.shape[1]]` without checking `if x.shape[1] > self.max_len:`. When `PosEmbed` is called directly as a submodule (`model["pos"](x)`) with `self.max_len == 1` and `x.shape[1] > 1`, `self.weight[:x.shape[1]]` has shape `[1, D]` and **silently broadcasts** position 0 across all `T` positions instead of raising `HNDLError` (unlike `Attention.forward` at `attention.py:314-317`, which explicitly raises `HNDLError("E_CONSTRAINT", ...)` when `T > max_len`).
     - *Fix:* Guard `if x.shape[1] > self.max_len:` in `PosEmbed.forward` and raise `HNDLError("E_CONSTRAINT", ...)`.
  3. **`MixtureOfExperts` (`src/hndl/operators/moe.py:21-38, 125-132`):**
     - `_reference` (lines 21–38) is an exact copy of `MixtureOfExperts.forward`'s sparse `nonzero` + `index_add` loop rather than an independent dense reference (`sum` over experts weighted by the top-k softmax mask), so `check_reference` does not independently verify the sparse routing index math.
     - `rows = mask.any(dim=-1).nonzero(as_tuple=True)[0]` + `if rows.numel() == 0:` forces a GPU-CPU synchronization per expert and breaks `torch.compile(fullgraph=True)` (requiring `moe_transformer` to be skipped in `tests/benchmark/test_compile_compat.py:25-28`), and `index_add` raises on CUDA when `torch.use_deterministic_algorithms(True)` is enabled.

---

#### [M-13] Missing `reference=` Implementations (20 Operators) and Missing Dedicated `tests/operators/test_<alias>.py` Files (15 Operators)
- **Component:** Operator Catalog & Test Suite (`src/hndl/operators/*.py`, `tests/operators/`)
- **Evidence:**
  1. **20 of 69 operators do not declare `reference=`** (causing 80 test skips in `tests/operators/test_all_operators.py:42`):
     `adaptive_norm`, `add`, `chunk`, `cls_token`, `concat`, `conv`, `coordinate_grid`, `deconv`, `flatten`, `fourier_features`, `grid_sample`, `group_norm`, `leaky_relu`, `linear`, `pos_embed`, `pretrained`, `relu`, `reshape`, `split`, `tanh`.
  2. **15 of 69 operators do not have a `tests/operators/test_<alias>.py` file** (despite `docs/ADDING_OPERATORS.md:15` stating that every operator has one):
     - 4 are tested in `tests/operators/test_style_renderer.py`: `broadcast_mul`, `coordinate_grid`, `fourier_features`, `grid_sample`.
     - 2 are tested in top-level `tests/`: `adaptive_norm` (`tests/test_adaptive_normalization.py`) and `pretrained` (`tests/test_pretrained.py`, `tests/test_pretrained_provider.py`).
     - **9 have neither a dedicated `test_<alias>.py` file nor a `reference=` declaration**: `add`, `concat`, `flatten`, `group_norm`, `leaky_relu`, `relu`, `reshape`, `split`, `tanh`.
- **Recommendation:** Add `reference=` functions to all 19 non-pretrained operators currently missing one (each is 1–4 lines of `torch` / `torch.nn.functional`), and add dedicated unit test files (or consolidate and document the mapping in `docs/ADDING_OPERATORS.md`).

---

#### [M-14] CI Workflow Gaps (`publish.yml`, `ci.yml`), Benchmark Suite Issues, and MkDocs Link Rewriting
- **Component:** CI/CD, Benchmarks & Docs (`.github/workflows/publish.yml:68-73`, `.github/workflows/ci.yml:3-7`, `tests/benchmark/`, `scripts/gen_doc_pages.py:38-50`)
- **Evidence & Issues:**
  1. **`.github/workflows/publish.yml:68-73`:** The step `"Install the wheel outside the checkout"` runs `cd "$RUNNER_TEMP"`, which only affects that subshell. The next step, `"Test the installed wheel"`, runs in `$GITHUB_WORKSPACE` (the repo checkout), where `pyproject.toml:50` sets `pythonpath = ["."]`, putting the source checkout on `sys.path`!
     - *Fix:* Run `cd "$RUNNER_TEMP"` inside the `"Test the installed wheel"` step and pass `"$GITHUB_WORKSPACE/tests"` to `pytest`, and add a step testing `pip install dist/*.tar.gz` or running `pytest` inside the unpacked sdist.
  2. **`.github/workflows/ci.yml:3-7`:** `on.push.branches` only lists `[master]`, even though day-to-day integration happens on `develop`.
     - *Fix:* Add `develop` to `on.push.branches: [master, develop]`.
  3. **`tests/benchmark/`:**
     - `tests/benchmark/conftest.py:42-44` defines a parametrized `device` fixture (`cpu`, `cuda:0`), but all three benchmark test files (`test_compile_compat.py:21`, `test_example_networks_latency.py:21`, `test_parity_vs_handwritten_pytorch.py:32`) hardcode `DEVICE = "cpu"` and never use the fixture.
     - `HandwrittenTransformerBlock` (`tests/benchmark/test_parity_vs_handwritten_pytorch.py:208-238`) uses `nn.MultiheadAttention` (which fuses Q, K, V into a single `in_proj_weight` GEMM), whereas HNDL's `TransformerBlock` uses `Attention` with 3 separate `nn.Linear` projections (`q_proj`, `k_proj`, `v_proj`), inflating the measured HNDL-to-PyTorch latency ratio on CPU (~1.38x–1.49x against a 1.5x ceiling) due to kernel choice rather than `GraphModule` overhead.
  4. **`scripts/gen_doc_pages.py:38-50` & `README.md`:** Because `tests/test_readme_links.py` requires all links in `README.md` to start with `https://`, `README.md` links to `https://hypergan.github.io/HNDL/...` with raw file paths as link text (e.g. `[docs/networks.md](https://hypergan.github.io/HNDL/networks/)`), and `rewrite_target` in `scripts/gen_doc_pages.py` skips all `https://` URLs, so local `mkdocs serve` navigation from the home page jumps out to production URLs.

---

### 3.3 Low-Severity Findings (`L-01` – `L-10`)

- **[L-01] Misleading Loop Collision Message for Auto-Generated Node Names (`src/hndl/config.py:97-103`):**
  When a loop-suffixed node name (e.g. `name="n"` in iteration 0 -> `"n0"`) collides with an earlier auto-generated node `n0`, `self.suffixed.get("n0")` is `None`, so `_Interpreter.node_name` formats the owner as `"an explicit name='n0'"` even though `n0` was auto-generated.
- **[L-02] Native Python Capture Emit Errors Omit `node=` Context (`src/hndl/capture.py:127-189, 282-305`):**
  In `resolve_callable`, `OperatorNamespace` calls `capture.emit(op, args, kwargs)` with `source=None`. If `emit` raises `E_ARGUMENT`, `E_BINDING`, or `E_POLICY_CONFLICT`, the resulting `HNDLError` has `node=None` even when the caller passed an explicit `name="my_node"`.
- **[L-03] `_Solver.patterns` Blames First Ellipsis Port on Rank Underflow (`src/hndl/resolver.py:374-387`):**
  `ellipsis_length` is computed from the first port in `entries` with `ELLIPSIS in pattern` and a known `shape`. In the second loop (line 386), if `ellipsis_length < 0`, `self.error` names the first port in `entries` with `ELLIPSIS`, even if its shape was `None` and a later port (`out`) had the invalid rank.
- **[L-04] Defensive Bounds Checks in `_Solver.axis`/`interval` and `hndl.relations` (`src/hndl/resolver.py:284-291, 327-341`, `src/hndl/relations.py:59-72, 82-95`):**
  `_Solver.axis` and `_Solver.interval` do not bounds-check `axis` before indexing `shape[axis]`; `conv_transpose_input(1, kernel=4, stride=1)` returns `-2` instead of `None` (should require `extent >= 1` and return `None` when `numerator // stride + 1 < 1`); `conv_input_range` with `extent <= 0` can return `lower <= upper` instead of an empty interval `(1, 0)`.
- **[L-05] `Example`/`Node` Post-Init Guards and Duplicate `argument_origins` (`src/hndl/operator.py:284-293`, `src/hndl/types.py:257-264`, `src/hndl/resolver.py:449-458`):**
  `Example.__post_init__` converts `input_shape` and `output_shape` with `tuple(...)`, which turns a multi-port dict `{"x": ("B", 4), "y": ("B", 4)}` into `("x", "y")`; `Node.__post_init__` converts `outputs="out"` into `('o', 'u', 't')` if a string is passed instead of a tuple; `ResolvedNode` stores `argument_origins` in both `node.source["argument_origins"]` and `node.provenance`.
- **[L-06] `_state_bytes` Meta-Device Estimation Uses `id(value)` Instead of Storage Identity (`src/hndl/torch.py:995-1013`):**
  When `compute_dtype is not None` (during the meta-device `max_state_bytes` pre-check), `_state_bytes` keys `storages` by `id(value)` rather than `value.untyped_storage()._cdata`, which would overcount if two parameters inside a custom module were views of the same meta storage.
- **[L-07] Unused `Chunk.size` Attribute and Positional Optional Flags on `GridSample` (`src/hndl/operators/chunk.py:141-144`, `src/hndl/operators/grid_sample.py:12-16`):**
  `Chunk.__init__` computes `self.size = None if dim == 0 else input_shapes["x"][dim] // chunks`, which is never read anywhere; `GridSample` leaves optional `mode`, `padding_mode`, and `align_corners` as `positional=True` (unlike `Upsample` and `Pad`).
- **[L-08] Missing `extra_repr()` and Direct-Init Validation Consistency (`src/hndl/operators/coordinate_grid.py`, `fourier_features.py`, `grid_sample.py`, `_equalized.py`, `attention.py`, `resblock.py`, `transformer_block.py`):**
  `CoordinateGrid`, `FourierFeatures`, and `GridSample` lack `extra_repr()`; `EqualLinear` and `Attention.extra_repr()` omit `equalized`; `ResBlock.__init__` does not validate `out_channels % groups == 0` on direct construction; `transformer_block._reference` duplicates attention and feed-forward reference code.
- **[L-09] Minor Documentation & Comment Typos (`docs/ADDING_OPERATORS.md:217`, `tests/benchmark/_timing.py:10`, `tests/benchmark/test_parity_vs_handwritten_pytorch.py:41`):**
  `docs/ADDING_OPERATORS.md:217` omits `scripts` from `ruff check src tests examples`; benchmark comments cite `"torch 2.14"` (non-existent PyTorch version).
- **[L-10] Git Worktree, Branch, and `.venv` Hygiene (`.claude/worktrees/`, `pyproject.toml:48-53`):**
  - `.venv/bin/*` console scripts have stale shebangs pointing to `/home/martyn/dev/hypergan/hndl/.venv/bin/python` and `.venv/lib/python3.14/site-packages/__editable__.hndl-0.8.0.pth` points to `/home/martyn/dev/hypergan/hndl/src` (an older directory location). Re-running `.venv/bin/python -m pip install -e '.[dev]'` will point `.venv` at `/home/martyn/dev/hndl`.
  - `git worktree list` has 28 prunable entries and 2 nested worktrees inside `.claude/worktrees/`, plus ~90 stale local `worktree-*`/`merge-pr*` branches. Adding `norecursedirs = [".claude", ".venv", "build", "dist"]` to `[tool.pytest.ini_options]` in `pyproject.toml` prevents accidental test collection inside `.claude/worktrees/`.

---

## 4. Prioritized Remediation Roadmap

### Phase 1: High-Priority Correctness & Packaging Fixes
1. **Fix `MANIFEST.in` (`H-07`):** Include `CHANGELOG.md` and `recursive-include tests *.py *.json` so `sdist` builds include the 0.6.0 plan fixtures and pass `pytest`.
2. **Fix `RMSNorm.forward` `float64` Precision Loss (`H-06`):** Only cast `x` to `float32` when `x.dtype in (torch.float16, torch.bfloat16)` in `src/hndl/operators/rms_norm.py:77`.
3. **Fix Parser `\r` and `NEWLINE` Screening Bugs (`H-01`, `H-02`):** Normalize `\r\n`/`\r` line breaks in `_parser.py` and reset `previous = None` on `tokenize.NEWLINE` in `_screen`.
4. **Fix `_is_chain` and Slice Step Validation (`H-03`):** Require `node.outputs == ("out",)` in `_is_chain` and reject `slice.step not in (None, 1)` in `GraphModule.__getitem__`.
5. **Fix `Pretrained` `trainable=True` Training Mode Sync (`H-04`):** Call `layer.train(layer.training)` after applying `requires_grad_` overrides in `src/hndl/torch.py:_build`.
6. **Fix `TimmProvider.instantiate` Revision Pinning (`H-05`):** Pass the resolved `@revision` when instantiating `hf://` timm checkpoints.
7. **Sync `SPEC.md` and `IMPLEMENTATION.md` to `0.8.0` (`H-08`):** Update version headers (`0.1.2` -> `0.8.0`), operator count (`59` -> `69`), `Arg(..., since=...)`, and initialization notes, and enforce version sync in `tests/test_package.py`.

### Phase 2: Medium-Priority Hardening & Test Coverage
1. **Complete Operator `reference=` & Test Suite Coverage (`M-12`, `M-13`):** Add `reference=` functions to the 19 non-pretrained operators currently skipped in `test_all_operators.py`, add missing operator tests, fix `attention._rope_tables` `float64` precision, add `T > max_len` validation to `PosEmbed.forward`, and replace `moe._reference` with a dense oracle.
2. **Harden Pretrained Cache & Error Wrapping (`M-01`, `M-02`):** Include file `st_mtime_ns` / `st_size` in local `resolve_source` lookups and revisions, add `_require("safetensors")` to `TimmProvider`, and wrap `config.json` IO/JSON errors in `E_PRETRAINED`.
3. **Harden Parser, Capture, and Solver Edge Cases (`M-03` – `M-07`):** Reject `p = ()` and Python 3.11 `f""` strings in `_parser.py`, reject falsy `input_dtype` and keyword/scheme external port names, and fix `E_AMBIGUOUS` interval reporting in `resolver.py:439`.
4. **Tighten CI & Benchmarks (`M-14`):** Fix `cd "$RUNNER_TEMP"` in `.github/workflows/publish.yml`, add `develop` to `.github/workflows/ci.yml`, and align `HandwrittenTransformerBlock` projections in `test_parity_vs_handwritten_pytorch.py`.
5. **Add Public Type Annotations (`M-11`):** Annotate public functions and replace `: object` dataclass field types so `src/hndl/py.typed` satisfies strict `mypy`/`pyright` checks.
