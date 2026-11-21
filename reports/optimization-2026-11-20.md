# HNDL correctness and optimization review

20 November 2026 · HNDL 0.11.0 · `develop` at `55aecf0`

**Fix execution parity first.** Changing the execution backend can silently change which inputs a model accepts, which hooks run, and what a standalone model trains. The first performance changes I would ship are rotary-table reuse during decoding and removal of the dense expert-count matrix in MoE. Both improved an entire workload with the same weights and outputs. For authors, inspecting one small recipe took 1.25 seconds in a fresh process.

## Correctness findings

### High priority · Compiled preparation drops the public tensor contracts

A float32 `relu()` network with contract `[B, 8]` rejects `[2, 7]` and int64 in eager and generated execution. After a valid warm call, `torch.compile(model.prepare(), backend="eager", fullgraph=True)` returned both invalid inputs as results with their supplied shape or dtype. The compiling branch binds names and calls the generated tensor program directly.

Emit traceable shape, batch, dtype and device checks at the public boundary, and verify output contracts for custom operators. Exercise valid and invalid calls on the same compiled wrapper. [Prepared forward](../src/hndl/prepared.py#L450)

### High priority · Rotary caches can break training after inference

On CPU and CUDA, run `attention(2, causal=True, rope=True)` in eval mode under `torch.inference_mode()`, then call `model.train()` and forward the same shape with gradients enabled. Autograd raises `Inference tensors cannot be saved for backward`. Standalone `rope(2)` has the same failure. The retained sine and cosine tensors were created in inference mode.

Allocate reusable tables as ordinary no-grad tensors or rebuild them when gradients require it. Test the inference → training → backward transition with the same model. [Attention tables](../src/hndl/operators/attention.py#L445) · [Standalone tables](../src/hndl/operators/rope.py#L275)

### High priority · MoE dispatch can bypass output-changing hooks

With four experts, top-4 routing and a forward hook returning zeros on every expert's `down`, CPU sparse dispatch invoked four hooks and produced zeros. Padded and automatic grouped dispatch invoked zero hooks and returned the original output. This can silently disable an adapter or a debugging hook.

Use hook-aware module calls when an expert or projection has custom behavior. Verify output-changing hooks and gradients under each dispatch mode. [MoE dispatch](../src/hndl/operators/moe.py#L250) · [Existing projection eligibility checks](../src/hndl/operators/_packing.py#L133)

### High priority · Standalone Python fails for 15 of 25 recipes

Of the 25 bundled recipes, 10 generated models constructed, loaded the HNDL state dict and matched the original forward. The other 15 raised `NameError`, including GPT, LLaMA and ResNet-18. The generator copies a class body and omits helpers such as `LayerNorm`, `_split_heads` and `normalize_stride`. Even a runnable `linear(8)` with zero initialization and `trainable=False` emitted 72 trainable, nonzero scalars.

Emit the built-ins' dependencies and construction settings, and fail at generation with the affected node when an operator cannot be emitted. Test standalone construction **before** loading weights as well as parity after loading them. [Class emission](../src/hndl/codegen.py#L127) · [Generated constructor](../src/hndl/codegen.py#L184)

### High priority · Weight tying disappears from the semantic digest and diff

Two biased 8→8 linears own 144 parameter scalars. Adding only `tie={"weight": "a.weight"}` to the second leaves 80. The two plans returned the **same semantic digest**, an empty modified-node list and a zero parameter delta. Tying is stored in source metadata, which semantic serialization omits.

Include canonical tie relationships in the semantic representation and diff, count shared parameters once, and provide a compatibility path for saved tied plans. Keep line numbers and other provenance outside the digest. [Serialization](../src/hndl/types.py#L348) · [Diff](../src/hndl/types.py#L604) · [Build-time tying](../src/hndl/torch.py#L1733)

### Medium priority · YaRN outputs depend on the previous sequence length

On two fresh `RoPE(heads=1, rope_scaling="yarn", rope_scale=4)` layers, warm one with 128 positions, then give both the same all-ones `[1, 32, 16]` input. Maximum absolute output difference was **2.814**. YaRN uses the requested length to calculate frequencies, while a cache hit slices the longest prior table.

Key length-dependent schedules by their effective frequency context and test long → short → long order. The normal fixed-length graph boundary narrows exposure. [Frequency calculation](../src/hndl/operators/rope.py#L73) · [Cache hit](../src/hndl/operators/rope.py#L283)

### Medium priority · The native loader requires write access to a deployed package

A native CPU MLP exported and loaded successfully and matched eager output. Copying its `.pt2` and adjacent `.json` into a read-only directory caused `load()` to raise `PermissionError` opening `.hndl-deployment.lock`. The loader also touches the manifest after loading.

Support standalone read-only packages while retaining checksum and environment verification. Keep locking and LRU updates for the managed, writable export cache. [Lock](../src/hndl/deployment.py#L289) · [Loader](../src/hndl/deployment.py#L505)

## Static estimates that affect capacity decisions

The profiler, sidecars and actual model parameter counts agreed on **all 25 shipped recipes**. The following valid configurations expose gaps outside that set. For cache comparisons, batch was one, the plan length and cache capacity were both 16, and the measurement used `InferenceCache.allocated_bytes` after the first token.

| Configuration | Reported | Observed | Cause |
| --- | ---: | ---: | --- |
| `layer_norm(affine=False)`, width 32 | 64 parameters | 0 | Profile reads `elementwise_affine`. The declared argument is `affine`. |
| Float32 plan, `cast("float64")`, output `[2,16]` | 128 output bytes | 256 | Every output is charged at the plan dtype. |
| Default unabsorbed latent attention, 4 heads, latent rank 8, content/rotary/value head widths 4/4/4 | 768 cache bytes | 3,072 | Estimate assumes compressed storage. The default path stores expanded per-head keys and values. |
| bfloat16 Mamba2, model width 16, 4 heads × 8, state width 4, conv width 4 | 448 cache bytes | 896 | Recurrent state accumulates in float32 and convolution history includes the B/C projection channels. |

Use resolved port dtypes, operator argument schemas and runtime storage layouts for these formulas. Expose the assumed cache capacity and absorption mode in the summary so users can budget a deployment. Cross-check each formula against a built module or allocated cache, including ties and reduced precision. [Parameter estimates](../src/hndl/profile.py#L201) · [Cache estimates](../src/hndl/profile.py#L657) · [Activation bytes](../src/hndl/profile.py#L772)

## Performance findings

### 1. Build rotary tables once per decoding capacity

With two causal 4-head transformer blocks, width 128, batch 1, a 1,024-token prompt and 32 generated tokens, the current table growth took **858 µs/token** on CPU. Pre-sizing each layer's table to 1,056 before prefill took **602 µs/token**, a 30% reduction. Median setup plus prefill was 20.38 and 19.60 ms respectively. Both runs matched the full-sequence output. Each new maximum currently rebuilds the prefix's trigonometric tables, even though each step consumes only the new row. [Attention decode](../src/hndl/operators/attention.py#L491)

Use bounded geometric growth or the declared session capacity for length-independent schedules. Charge retained bytes to the appropriate owner and preserve captured table lifetimes across CUDA graph replay. Include a fresh-prefix benchmark. The existing benchmark computes its oracle on the same model at the final length *before* timing, which populates these tables for every measured token. [Existing benchmark](../tests/benchmark/test_prepared_execution.py#L185) · [Replay lifetime test](../tests/test_prepared.py#L365)

### 2. Compute MoE group offsets from sorted routes

Grouped dispatch sorts route IDs, materializes a `[routes, experts]` int64 one-hot matrix, sums it and takes a cumulative sum. Each required offset is already the right insertion point of the expert ID in the sorted array. A local substitution using `torch.searchsorted` produced the following results for an unpacked MoE with 4,096 tokens, width 32, 128 experts, hidden width 64, top-2 routing and float32 weights. [Routing implementation](../src/hndl/operators/moe.py#L264)

| Measurement | Current | Sorted offsets |
| --- | ---: | ---: |
| CPU offsets only, 8,192 sorted routes × 128 experts | 257.48 µs | 4.31 µs |
| CPU entire layer forward, including router and projections | 5.38 ms | 4.84 ms |
| CUDA entire forward, peak allocated above pre-call baseline | 29.39 MiB | 12.39 MiB |

The dense count tensor alone is 8 MiB. The complete CPU layer improved by 10%. The substitution matched outputs on both devices and matched input and parameter gradients on a smaller run with empty experts. For grouped dispatch, right insertion points give exclusive end offsets. Padded dispatch needs start offsets. Qualify both conventions and supported compiler/device combinations before sharing the implementation.

```python
experts = torch.arange(len(self.experts), device=expert_ids.device)
ends = torch.searchsorted(expert_ids, experts, right=True, out_int32=True)
```

### 3. Make checkpointing effective at the first trainable block

For `attention(4, causal=True)`, input `[1,32,32]`, trainable weights and ordinary inputs with `requires_grad=False`, both `checkpoint=True` and its generated equivalent saved **10 tensors totaling 37,376 logical bytes** during forward. Applying non-reentrant checkpointing directly to the same layer saved **one tensor, 4,096 bytes**, and produced matching parameter gradients. HNDL gates the call on an input requiring gradients and skips checkpointing this common first-block case. Use training and gradient mode when honoring a configured checkpoint, and exercise the compiled path, which also has its own dispatch. These counts come from saved-tensor hooks. Actual device peak and recomputation time need separate measurements. [Checked forward](../src/hndl/torch.py#L968) · [Warm eager](../src/hndl/torch.py#L1082) · [Generated forward](../src/hndl/prepared.py#L94)

### 4. Bound local-attention KV storage by the window

A float32 causal attention layer with width 128, 4 KV heads, session capacity 4,096 and window 64 retained **4 MiB** on the first token. Attention slices its visible keys *after* allocating full-capacity K and V. A 64-position persistent history at the same width is **64 KiB per layer**. A ring buffer could reduce retained history by 64× on this shape. Multi-token prefill can still require temporary workspace. Test chunks shorter than, equal to and longer than the window, wraparound, and absolute RoPE positions against full causal attention. [KV allocation](../src/hndl/inference.py#L281) · [Window selection](../src/hndl/operators/attention.py#L495)

### 5. Reduce small-model overhead where users feel it

On a warmed five-node MLP (64→128→128→10, two ReLUs), the same layers and inputs gave the following one-thread CPU latencies. Inference used five rotating rounds. The training rows use the repository benchmark, which includes gradient reset, forward, loss and backward.

| Workload | Eager HNDL | Prepared HNDL | PyTorch Sequential |
| --- | ---: | ---: | ---: |
| Inference, batch 1 | 29.10 µs | 23.57 µs | 17.58 µs |
| Inference, batch 32 | 49.35 µs | 43.04 µs | 36.36 µs |
| Training, batch 1 | 202.07 µs | 190.84 µs | — |
| Training, batch 32 | 257.86 µs | 247.30 µs | — |

There is about **6 µs per call** left between prepared execution and the same sequential layers at batch 1. Profile binding, signature formation and output unpacking before adding a new execution backend. For decoding, the full model parameter/buffer signature alone cost **71.65 µs** on the two-block workload above. Precompute a flat traversal at bind/rebind and measure the remaining per-tensor checks against mutations, buffer replacement and device moves. Removing version checks would let an active session mix weights from two checkpoints. [Prepared call](../src/hndl/prepared.py#L464) · [Session signature](../src/hndl/inference.py#L37)

## Authoring and contributor ergonomics

- **Shorten the first CLI invocation.** Fresh-process medians were 53 ms for `import hndl`, 60 ms for `hndl --help`, 1,251 ms for `inspect examples/networks/mlp.hndl --profile`, and 1,365 ms for `check examples/networks` across all 25 recipes. Once imports were warm, resolving the five-node MLP took 0.565 ms and building its resolved plan took 1.064 ms. Built-in discovery imports every operator implementation and therefore PyTorch on first resolve. Batch checks already amortize this cost. For editor/CLI use, prototype loading schema and relation metadata independently of module construction and compare cold startup as well as warm resolve. [Discovery](../src/hndl/operators/__init__.py#L16) · [Registry](../src/hndl/registry.py#L49)
- **Resolve the `cache` name collision.** A supported rank-2 input named `cache` resolves and executes positionally. The equivalent `model(cache=tensor)` fails with `E_BINDING` because Python binds the tensor to the session argument. Preserve keyword access for previously saved plans that declare this port and give new ambiguous declarations an actionable diagnostic or a separate session entry point. [Reservations](../src/hndl/resolver.py#L45) · [Network binding](../src/hndl/torch.py#L1273)
- **Make CLI options meaningful.** On the MLP recipe, both `inspect --profile --device cpu` and `--device nonsense` exited successfully with identical output. Static profiling never reads the option. Deprecate the flag or give it a documented effect once target-specific estimates exist. [CLI argument](../src/hndl/__main__.py#L146)
- **Make local checks independent of Git housekeeping.** The sole failure in the CPU suite came from an existing merged branch named `fix/provider-shape-callbacks`. The assertion checks the developer's local refs. Move branch/worktree hygiene to an explicit maintainer check so a contributor can run the product suite in a normal clone. [Failing test](../tests/test_packaging_stress.py#L391)

## Verification and reproduction

Measurements used the checkout's `src` tree, Linux x86-64, an AMD Ryzen 9 5900X, Python 3.14.7 and PyTorch 2.14.1+cu130. CPU timings use one intra-op thread. Physical GPU 1 was an RTX A6000. Both GPUs had other active workloads, so the fresh CUDA comparisons above cover parity and this process's allocated bytes. For historical CUDA latency, including replay, exclusive native loading and packing, see the [0.11.0 measurements](../docs/performance.md).

| Check | Result |
| --- | --- |
| CPU suite with CI's `not network and not benchmark` selection | 4,555 passed, 590 skipped, 139 deselected, 1 local-branch assertion failed |
| Prepared execution, packed execution, incremental decoding and session guards with CUDA visible | 204 passed, 10 skipped across CPU/CUDA cases |
| Opt-in CPU prepared inference, training and incremental benchmarks | 7 passed. At width 128 and prefix 1,024, the warmed benchmark measured 0.633 ms/token cached and 21.35 ms/token recomputing the prefix. |
| Lint, generated doc/stub freshness, recipe validation | Ruff passed, generated files current, 25/25 recipes resolved and profiled |
| Actual CPU native package | First export 8.26 s, artifact hit 20.2 ms, loaded outputs matched eager. One observation each. |

Run the project checks from the repository root with a development install.

```sh
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' python -m pytest -q -m 'not network and not benchmark'
CUDA_VISIBLE_DEVICES='' HNDL_BENCHMARK_MIN_TIME=0.6 python -m pytest -q -s \
  -m benchmark tests/benchmark/test_prepared_execution.py \
  -k 'inference_setup or training_forward or incremental_steps'
CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/test_prepared.py \
  tests/test_packed_execution.py tests/test_incremental_decoding.py \
  tests/test_inference_session_guards.py
python -m hndl check examples/networks
python -m hndl.docs --check
ruff check src tests examples scripts
```

**Reproducing the comparisons.** For standalone export, iterate `names()` and `load(name)` in the recipe test loader, resolve and build each CPU plan, then execute `plan.to_python_source()` in an empty namespace. Instantiate `Model().eval()`, load the eager state dict strictly, and compare batch-1 outputs under `no_grad()` with `random_inputs(plan, batch_size=1)`. [Recipe test loader](../tests/_network_examples.py)

For the rotary experiment, use the repository decoder fixture with `prefix=1024`, `steps=32`, `width=128` and initialization seed 1. Compute the full-sequence oracle, then clear each attention layer's `_rope_cache` before **every** prefill in both cases. In the pre-sized case call `layer._get_rope_tables(1056, tokens.device, tokens.dtype)` just before prefill. Use a fresh `InferenceCache(1056)` for each burst, time prefill and the 32 single-token calls separately, alternate the order, discard the first pair, and take the median of seven pairs. Compare concatenated outputs with the oracle at `rtol=1e-4, atol=1e-5`. [Decoder fixture](../tests/benchmark/test_prepared_execution.py#L172)

For MoE, compare the stock grouped forward and the offset substitution above on the same layer and sorted route IDs. CPU timings are medians of three alternating `torch.utils.benchmark.Timer.blocked_autorange` rounds. For CUDA memory, warm both paths, synchronize, reset peak allocator statistics, run one forward and subtract pre-call `memory_allocated()` from `max_memory_allocated()`. CLI numbers include interpreter startup and exit, using the median of five fresh processes with warm filesystem caches.
