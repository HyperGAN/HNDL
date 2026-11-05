# Incremental decoding

A causal sequence model can process a prompt once and then append only the new
positions. `InferenceCache` keeps attention keys and values, recurrent states,
and the short convolution histories between calls. The result of concatenating
the outputs of the chunks matches a full causal forward pass, within the
numerical tolerance of the selected dtype.

```python
import torch
from hndl.inference import InferenceCache
from hndl.torch import network

model = network(
    'attention(4, causal=True, rope=True)',
    input_shape=('B', 32, 64),
    output_shape=('B', 32, 64),
    device='cpu',
).eval()
cache = InferenceCache(max_seq_len=128)

with torch.no_grad():
    prompt_output = model(torch.randn(2, 32, 64), cache=cache)
    next_output = model(torch.randn(2, 1, 64), cache=cache)

cache.reset()  # Start an independent sequence.
```

Use `eval()` together with `torch.no_grad()` or `torch.inference_mode()`.
Cached state belongs to one model, batch, feature shape, dtype, device, and
parameter version. Reset the cache after changing any of them. Autocast mode
must also stay fixed. Each session needs its own cache; concurrent or recursive
reuse of a cache is unsupported. If a graph call fails after modifying state,
reset the cache before retrying.

`attention` and `transformer_block` require `causal=True`. Grouped-query
attention, query/key normalization, sliding causal windows, and logit softcaps
work with incremental decoding. Queries in a later chunk attend to their
absolute positions in the cached prefix. Rotary embeddings use the absolute
position offset too; they do not restart from zero on each chunk.

`latent_attention` also requires causal attention. Its normal mode stores
expanded keys and values. Calling `absorb_weights()` on an unbiased module
before starting the session uses compressed latent vectors and shared rotary
keys instead. Changing absorption mode requires a fresh session.

`mamba2` and `gated_delta_attention` retain a recurrent matrix and the raw inputs
needed by their causal depthwise convolutions. Their state size remains
constant as the sequence grows. `pos_embed` and standalone `rope` retain their
position counters, so they can appear in an incremental model as well.

Attention buffers allocate their full `max_seq_len` capacity on first use.
Appending a token writes its slot without copying the existing prefix. An
append past the capacity raises an error; create a larger cache or reset to
start a new sequence. A sliding window currently bounds attended positions,
while storage still uses the declared sequence capacity. The
`cache.allocated_bytes` property reports bytes retained by built-in tensor
states. `cache.position(node_id)` reports the number of positions consumed by
an operator; a transformer block uses the child key `node_id + '/attention'`.

The cache retains the existing dictionary interface: `get`, `set`, indexing,
`keys`, `values`, `items`, and `reset(node_id)`. Resetting a composite node also
clears its child states. A graph session requires a complete `reset()` after
clearing one of its nodes, so layers cannot silently lose part of the prefix.
When using layers directly, supply a unique `node_id`
to each cache-aware layer.

Spatial relative-position attention, spatial RoPE, and YaRN's current
sequence-length-dependent frequency schedule cannot use an incremental cache.
Length-independent RoPE modes (`none`, `linear`, `ntk`, and `llama3`) support
absolute offsets. Stateful cache mutation is currently eager-only; compile the
ordinary full-sequence path separately.
