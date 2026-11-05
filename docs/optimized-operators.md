# Packed projections and specialized attention

Prepared execution can combine projections and select fused attention kernels:

```python
import torch
from hndl.torch import network

model = network(
    'attention(4, causal=True, window_size=32, logit_softcap=10.0)\nswiglu(256)',
    input_shape=("B", 64, 128),
    output_shape=("B", 64, 128),
    device="cuda:0",
).eval()
x = torch.randn(2, 64, 128, device="cuda:0")
fast = model.prepare(pack=True, attention_backend="auto")
with torch.inference_mode():
    y = fast(x)
```

Packing is optional. Existing `Parameter` objects, their names and shapes,
`state_dict` keys, and references held by an optimizer remain valid. Preparation
places eligible parameter objects into contiguous shared storage, so their data
addresses change. In-place optimizer updates and checkpoint loads immediately
update the packed views. Packed execution supports ordinary backpropagation and
higher-order input gradients.

Self-attention and gated delta attention can combine Q/K/V projections into
one matrix multiplication; cross-attention combines the context's K/V projections, and SwiGLU combines its
gate/up projections. Different grouped-query projection widths are supported.
Individual projection hooks retain their normal calls. Tied parameters and
distinct parameters that already share storage retain their original layout.
Moves and dtype casts release cached old-device views and rebuild layouts on the
next call. Make a new prepared executor after changing the model's structure.
The prepared executor uses the root's mutation epoch and registration counter,
plus cheap parameter identity checks, to reuse packed layouts without scanning
tensor storage metadata on each call.
Direct edits to registered dictionaries or replacement of a parameter's
underlying storage require `model.rebind_state()` before reusing it. Ordinary
optimizer updates and in-place weight edits need no rebind.
Packed training also preserves the version checks that reject changing a
weight between forward and backward when it is needed for an input gradient.
Functional parameter swaps use differentiable parameter assembly and leave
the original packed storage intact.

## Sparse experts

Mixture-of-experts execution prefers supported grouped matrix multiplication.
Its portable fallback balances latency against temporary memory: small padded
batches remain available, while an estimated padding cost above 16 MiB selects
sparse execution. On CPU that path runs only the tokens assigned to each expert;
CUDA and compiler tracing use bounded tiles of selected matrices. Backward
gathers selected weights again, so it does not retain a separate matrix for
every route. Temporary matrix gathers target at most 1,048,576 elements per
tile, or one projection matrix when that matrix alone is larger. Parameter and
gradient storage still scale with the number of experts.

To select the memory tradeoff explicitly, call the operator module's
`configure_dispatch("sparse")` or `configure_dispatch("padded")` method.
`configure_dispatch("auto", max_padding_bytes=...)` changes the automatic
fallback threshold. This estimate covers padded activations rather than a
limit on the whole model's memory. Sparse fallback can be slower than padded
matrix multiplication for small expert counts and large token batches.

With packing enabled, expert-major parameter tensors share the original expert
parameters' storage and avoid per-call parameter stacking. Supported PyTorch
versions/devices use grouped matrix multiplication for aligned projections.
Other devices, dtypes and layouts retain portable dispatch. Compiler tracing
uses differentiable parameter assembly instead of the eager storage cache.

## Fused window and softcap attention

`attention_backend="auto"` tries PyTorch FlexAttention for supported CUDA
float32, float16 and bfloat16 inputs. It applies softcapping inside the attention
kernel and represents sliding windows with a block mask, avoiding a dense score
tensor or dense sliding-window mask in these paths. Ordinary attention keeps
using PyTorch scaled-dot-product attention.

The automatic backend retains the existing implementation for attention
dropout, dense additive bias such as learned relative-position bias,
unsupported head dimensions, older GPUs, and CPU execution. The currently
supported fused head width is a multiple of eight from 16 through 128, and matches the
value width. `attention_backend="reference"` disables the optional backend;
`"flex"` requests it explicitly and reports unsupported specialized calls.
Enclosing `torch.compile` or export tracing uses the existing traceable tensor
operations; it does not invoke the eager FlexAttention dispatcher.

Self-attention uses this dispatcher before constructing any dense window mask,
including during incremental decoding. Cached queries keep their absolute
position offsets; causal window reads trim the irrelevant prefix before kernel
dispatch. The cache retains its fixed sequence capacity.

Each attention module retains at most sixteen block masks. Casting or moving a
module clears these masks. The module's `_hndl_attention_backend` attribute
records `"flex"` or `"reference"` after execution, and
`_hndl_attention_fallback_reason` describes an automatic fallback. The first
fused invocation compiles kernels, so measure warmed latency separately from
startup and retained memory.
