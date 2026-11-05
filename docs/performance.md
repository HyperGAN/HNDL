# Performance measurements

HNDL provides several execution paths because setup cost, steady-state latency,
training, and retained memory have different tradeoffs. The measurements below
exercise the public prepared, incremental, and native deployment interfaces.
They are observations from one shared workstation, with numerical parity
checks, rather than portable speed guarantees.

## Reproduce the measurements

```sh
OMP_NUM_THREADS=1 HNDL_BENCHMARK_MIN_TIME=0.3 \
  python -m pytest -m benchmark tests/benchmark/test_prepared_execution.py -q -s
```

The module prints `HNDL_PERF` JSON records. It is opt-in and excluded from the
ordinary CI selection. Absolute timings are not pass/fail thresholds. The
checks cover output and training-gradient parity, cache limits, repeated graph
eviction, and deployment cache reuse.

The prepared, training, replay-memory, incremental, and native-deployment
measurements were rerun against HNDL 0.11.0 release candidate `2ac936e` in
November 2026. They used Python 3.14.8, PyTorch 2.14.1+cu130, an AMD Ryzen 9
5900X, one intra-op CPU thread, and an NVIDIA RTX A6000. Physical GPU 1 was exposed as
`cuda:0` with `CUDA_VISIBLE_DEVICES=1`. Inputs remained on their compute device;
host transfers and framework imports are excluded. Cold CUDA compilation used
CUDA 13.0 runtime/CRT/CCCL headers and libraries, plus CPython 3.14 development
headers through an include path. The development-header package was 3.14.10;
the running interpreter was 3.14.8.

Warm timings use `torch.utils.benchmark.Timer`, three interleaved rounds, and
the median of the round medians. Stateful decoding uses three finite token
bursts with explicit CUDA synchronization around each burst. Setup figures are
single observations. CPU and CUDA rows, and the separate operator window at
the end, should be compared within their own paired measurements.

The [benchmark source](https://github.com/HyperGAN/HNDL/blob/master/tests/benchmark/test_prepared_execution.py)
and [raw records](https://github.com/HyperGAN/HNDL/blob/master/reports/performance-prepared-execution.json)
record the workloads and results.

## Prepared inference

The MLP is `linear(128); relu(); linear(128); relu(); linear(10)`, with 64 input
features and float32 parameters. Every path shares the same parameters.
Inference runs inside `torch.inference_mode()`.

| Device | Batch | Eager, µs | Generated, µs | Replay, independent output, µs | Replay, borrowed output, µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | 39.0 | 29.2 | — | — |
| CPU | 32 | 107.2 | 53.5 | — | — |
| CUDA | 1 | 93.5 | 74.5 | 55.0 | 38.8 |
| CUDA | 32 | 117.4 | 96.4 | 56.9 | 51.4 |

Replay timings include the device input copy, runtime guards, stream ordering,
and the output clone when requested. Borrowed outputs are overwritten by later
replay of the same entry. The generated path retains ordinary PyTorch module
calls, hooks, gradients, and parameter references.

| Device | Batch | Generated setup, ms | Replay setup with clone, ms | Replay setup with borrowed output, ms |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 5.06 | — | — |
| CPU | 32 | 0.39 | — | — |
| CUDA | 1 | 2.31 | 6.31 | 3.81 |
| CUDA | 32 | 2.37 | 4.15 | 5.16 |

Setup starts with a built model and warmed device. The first preparation also
loads the prepared backend. Its executor class is cached by architecture, so
later rows can reuse its Python code.
CUDA setup includes warmup and capture. See [prepared execution](prepared-execution.md)
for backend selection and output ownership.

## Training

The same MLP runs in training mode. A step includes
`zero_grad(set_to_none=True)`, forward, mean-square loss, and backward. Inputs
do not require gradients; parameters do. There is no optimizer update.

| Device | Batch | Eager, µs | Generated, µs | Generated / eager |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 255.4 | 242.3 | 0.95 |
| CPU | 32 | 291.2 | 284.5 | 0.98 |
| CUDA | 1 | 553.9 | 533.8 | 0.96 |
| CUDA | 32 | 636.6 | 589.4 | 0.93 |

Here most work remains ordinary PyTorch forward/backward execution. CUDA replay
is an inference path and is not selected for these training calls.

## Replay memory

The batch-1 entry reported 2,097,408 retained bytes: a 2 MiB private graph pool
plus its input buffer. The first capture increased the process's reserved CUDA
memory by 22 MiB. A later entry with the shared framework workspace already
warm added 2 MiB. cuBLAS workspace and allocator granularity explain why total
process reservation differs from the per-entry graph budget.

With `max_graphs=1` and an 8 MiB graph budget, six successive batch signatures
kept one graph and a flat 22 MiB process-reserved delta. Clearing the cache,
collecting Python objects, and emptying unused allocator blocks returned to
the measurement baseline in that run. Cache limits bound retained graph
pools, input buffers, and pinned derived tensors; the shared CUDA framework
workspace has a separate lifetime.

## Incremental decoding

The decoder has two causal, four-head transformer blocks with RoPE, batch 1,
float32 weights, and the listed hidden width. Each measurement generates 16
tokens after the prefix. Prefill is outside the timed burst. The baseline
recomputes every growing prefix through the generated tensor program, without
the fixed-T HNDL boundary checks. Cached execution includes session and
parameter guards. Both paths use identical weights and match outputs.

| Device | Width | Prefix tokens | Recompute per token, ms | Cached per token, ms | Retained KV state, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 128 | 256 | 3.877 | 0.759 | 0.53 |
| CPU | 128 | 1024 | 20.977 | 0.789 | 2.03 |
| CPU | 512 | 1024 | 159.702 | 4.141 | 8.13 |
| CUDA | 128 | 256 | 0.692 | 0.989 | 0.53 |
| CUDA | 128 | 1024 | 1.287 | 1.091 | 2.03 |
| CUDA | 512 | 1024 | 3.543 | 1.297 | 8.13 |

The small CUDA decoder is dominated by dispatch and cache-management work;
caching is slower at the shorter prefix in this setup. The larger decoder
amortizes that overhead. The cache allocates for exactly prefix length plus
16 tokens and never rewinds positions during timing. See
[incremental decoding](incremental-decoding.md) for session ownership and limits.

## Native deployment

The batch-1 MLP is exported with a fresh Inductor cache. Python/PyTorch imports,
device initialization, and model construction are outside the export timer.
An artifact-cache hit still captures and fingerprints the model, while reusing
native compilation. Package loading is measured separately.

| Device | First export, s | Artifact hit, ms | Load, ms | Package, MiB |
| --- | ---: | ---: | ---: | ---: |
| CPU | 8.43 | 31.6 | 4.0 | 1.61 |
| CUDA | 6.00 | 30.4 | 10.2 | 1.70 |

| Device | Eager, µs | Default native runner, µs | Exclusive native runner, µs |
| --- | ---: | ---: | ---: |
| CPU | 43.7 | 31.9 | 32.2 |
| CUDA | 93.3 | 1100.5 | 82.6 |

On this tiny CUDA workload, synchronization in the default native runner
dominates the computation. `load(artifact, run_single_threaded=True)` avoids
that native pool path on supporting PyTorch versions. HNDL rejects overlapping
host calls and changes from the first CUDA stream in this mode, without adding
a per-call GPU synchronization. The default runner retains its ordinary
concurrent-use behavior. Output independence is checked for both modes.
See [native deployment](deployment.md) for contracts and supported loaders.

## Packed operators and fused attention

These are separate measurements on physical GPU 0, also an RTX A6000, using
float32, one CPU thread, three interleaved warmed rounds of 0.07 seconds per
case, and identical weights. They time raw operators, excluding the prepared
network wrapper. Packing uses a retained graph as the trusted owner. Training
includes input gradients, gradient reset, forward, mean-square loss, and
backward; there are no optimizer steps.

| Operator workload | Plain inference, µs | Packed inference, µs | Plain training, µs | Packed training, µs |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU `[1,128]`, hidden 512, no bias | 61.2 | 68.1 | 480.9 | 468.9 |
| SwiGLU `[512,256]`, hidden 1024, no bias | 119.9 | 111.1 | 657.5 | 617.6 |
| MoE `[256,64]`, 64 experts, hidden 128, top-2 SwiGLU | 4765.6 | 4248.8 | 18149.4 | 20993.6 |

The MoE uses biases, softmax routing, scaling 1, and automatic dispatch, which
selects eligible grouped matrix multiplication on this setup. Packing is
optional: small inference and this MoE training case became slower.

For Q/K/V `[1,4,512,64]`, a causal window of 64 and softcap 20, fused
FlexAttention measured 153.0 µs versus 202.8 µs for the reference with an
already cached dense mask. There is no dropout. The first fused compilation is
excluded. Ordinary attention remains on SDPA. The
[operator guide](optimized-operators.md) explains supported cases and fallbacks.
