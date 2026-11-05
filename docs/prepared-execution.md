# Prepared execution

HNDL can generate a straight-line PyTorch executor from an existing model. The
executor shares its layers and parameters: existing optimizers, layer hooks,
autograd, activation checkpointing and checkpoint names continue to work.

```python
import torch
from hndl.torch import network

model = network("linear(128); silu(); linear(10)",
                input_shape=("B", 64), output_shape=("B", 10), device="cpu")
fast = model.prepare()
x = torch.randn(8, 64)
loss = fast(x).square().mean()
loss.backward()
```

The first call for an input shape, dtype, device and training mode runs
the checked executor. Subsequent calls use generated tensor locals and release
intermediates after their last consumer. Preparation does not copy or initialize
weights. The input binding and output conventions match the original model.
An optional tuple or mapping of example inputs warms the executor immediately:
`fast = model.prepare((x,))` or `model.prepare({"x": x})`.

Preparation fixes module structure and Python hooks. Use
`model.allow_state_mutation()` or call `model.rebind_state()` after direct
structural edits, parameter storage replacements or changes to hooks. Moving or
casting the model invalidates preparations automatically. Ordinary optimizer
updates to existing parameters remain visible. The generated executor is
available as `fast._export_module` for compiler integrations; the normal
`torch.compile(fast, fullgraph=True)` path also traces its tensor operations.

## CUDA graph inference

Repeated inference with stable shapes can replay a captured CUDA graph. This
removes Python dispatch between operators and uses persistent input and output
buffers.

```python
model = model.to("cuda:0").eval()
x = x.to("cuda:0")
with torch.inference_mode():
    fast = model.prepare((x,), backend="cuda_graph", max_graphs=4,
                         max_graph_bytes=256 * 1024 * 1024)
    y = fast(x)
```

Each distinct input signature, including strides and gradient mode, gets a separate graph. An LRU cache limits both
the number of graphs and their retained CUDA allocations. A graph larger than
the byte budget raises `E_RESOURCE`; evicted graphs release their owned buffers.
`fast.cache_info` reports the retained graph count and byte accounting;
`fast.clear_cache()` releases all entries. The byte budget counts graph-owned
private pools, including inactive captured intermediates, and persistent input
buffers. Model weights, caller-owned output clones, shared CUDA library
workspaces and PyTorch's ordinary allocator cache are outside that budget.
Capture uses one shared stream per CUDA device; library warmup has a fixed
framework footprint rather than requiring a new stream for each cache entry.
PyTorch's ordinary allocator may retain freed memory for later reuse.

Outputs are cloned by default so earlier results remain independent of later
calls. Set `clone_outputs=False` only when the consumer finishes before the next
replay: these borrowed outputs are overwritten by a later call for the same
signature. Calls through one executor are serialized; use separate executors
for independent concurrent sessions.

CUDA replay requires `eval()`, disabled gradients and autocast, and inputs on
one CUDA device. Cast the model and floating inputs explicitly for reduced
precision. Python hooks cannot run on replay, so hooked graphs require generated
execution. Custom operators and pretrained wrappers require an explicit
`_hndl_cuda_graph_safe = True` declaration on their module after their capture
behavior has been verified. Stateful decoding uses the separate inference cache
API and generated execution rather than CUDA replay.

`backend="auto"` measures generated execution and CUDA replay on the first
eligible inference call for each signature, including input copies and the
chosen output ownership policy. It retains the faster strategy and falls back
to generated execution when capture is unsupported or exceeds the memory budget.
CPU and training calls use generated execution. This opt-in tuning adds warmup
and measurement work to the first call.

## Measuring a preparation

Measure startup separately from warmed execution. For CUDA, synchronize around
wall-clock measurements or use `torch.utils.benchmark.Timer`. Compare identical
weights, shapes, input residency and output ownership; a replay returning a
borrowed buffer does less work than a call returning an independent tensor.
Check training throughput and memory separately from inference latency.
