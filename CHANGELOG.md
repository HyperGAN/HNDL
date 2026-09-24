# Changelog

## Unreleased

- Add `broadcast_mul`, `coordinate_grid`, `fourier_features`, and `grid_sample`
  for style-conditioned coordinate renderers composed in HNDL. Fourier tables
  and coordinate grids are persistent buffers; sampling follows PyTorch semantics.

- Add opt-in `equalized=True` to `linear`, `attention`, and `feed_forward`:
  raw N(0,1) weights, zero biases, and runtime `1/sqrt(fan_in)` weight scaling
  with unit gain and learning-rate multiplier. Existing defaults, shapes and
  parameter names stay unchanged. Initializer overrides target raw weights;
  equalized linear rejects combination with spectral normalization.

## 0.6.0 (2026-09-21)

A minor release adding TransGAN-style relative-position-bias attention and
selectable random initializer schemes, plus a forward-pass performance pass.

- **`init=` selects random initializer schemes, not just constants.** An entry
  in the per-parameter `init` mapping may now be one of eight keyword-only
  scheme calls instead of a number, mixed freely with constants in the same
  mapping:

  ```python
  linear(init={"weight": xavier_uniform(gain=1.0), "bias": 0.0})
  attention(4, init={"relative_position_bias_table": truncated_normal(std=0.02)})
  conv(64, kernel_size=3, init={"weight": kaiming_normal(mode="fan_out", nonlinearity="relu")})
  ```

  The schemes are `xavier_uniform(gain=1.0)`, `xavier_normal(gain=1.0)`,
  `kaiming_uniform(a=0.0, mode="fan_in", nonlinearity="leaky_relu")`,
  `kaiming_normal(a=0.0, mode="fan_in", nonlinearity="leaky_relu")`,
  `truncated_normal(mean=0.0, std=1.0, a=-2.0, b=2.0)`,
  `normal(mean=0.0, std=1.0)`, `uniform(a=0.0, b=1.0)` and
  `orthogonal(gain=1.0)` --- each one the `torch.nn.init` function of the same
  name, with torch's own keyword names and defaults, so
  `truncated_normal(std=0.02)` is
  `torch.nn.init.trunc_normal_(parameter, mean=0.0, std=0.02, a=-2.0, b=2.0)`.
  `mode` is `fan_in` or `fan_out` and `nonlinearity` is one of the names
  `torch.nn.init.calculate_gain` accepts; every other keyword is a number
  rounded to float32 exactly like a constant override. Fills happen inside the
  construction RNG scope, in module-parameter order and then node order, so
  the same plan and the same `initialization_seed` reproduce them bit for bit.
  Xavier, Kaiming and orthogonal need a parameter of at least two dimensions:
  `linear(2, init={"bias": xavier_uniform()})` is `E_INITIALIZATION` naming
  the parameter and its shape, checked before any node is initialized rather
  than surfacing a raw `torch` `ValueError`.

- **The same initializer names work in native Python.**
  `from hndl import xavier_uniform` (or `from hndl.initializers import ...`)
  gives plain functions returning the record the declarative parser builds, so
  `ops.linear(init={"weight": xavier_uniform(gain=1.0)})` and
  `linear(init={"weight": xavier_uniform(gain=1.0)})` resolve to the same
  `semantic_digest`. The eight names are reserved: registering an operator
  alias that shadows one is `E_REGISTRY`, and a parse request carrying one is
  `E_NAME`.

- **Grammar: calls stay out of literal containers, with one narrow exception.**
  An initializer call is legal only as a direct value of an `init={...}`
  mapping on an operator call. A bare `init=xavier_uniform()`, a scheme in a
  list, a scheme nested inside another scheme's arguments, a scheme in
  `trainable={...}` or in any other keyword's dictionary, and any
  non-initializer call inside a literal container all remain `E_SYNTAX`,
  rejected by the isolated AST validator before any operator lookup happens.

- **Plan format: `initialization.kind` gains `torch_default@2`.** A node with
  at least one scheme override records
  `{"kind": "torch_default@2", "overrides": {"weight": {"kind": "xavier_uniform", "gain": 1.0}, "bias": 0.0}}`.
  Constant-only nodes --- including the empty default --- keep emitting
  `"torch_default@1"` unchanged, so every plan written before this release
  keeps its exact bytes, semantic digest and artifact digest. Both versions
  are accepted on restore forever; a scheme record under `"torch_default@1"`,
  a `"torch_default@2"` record with no scheme, a missing or extra scheme
  keyword, and a non-canonical float32 scheme argument are each
  `E_INITIALIZATION`.

- **`attention` learns a 2D relative-position bias.** With
  `attention(4, spatial_shape=(16, 16), relative_position_bias=True)` the
  `[B, T, D]` sequence is read as a row-major `H x W` token grid --- token `t`
  sits at `row = t // W`, `col = t % W` --- and each head learns one bias per
  query-minus-key offset, the Swin/ViT/TransGAN construction. The learned
  parameter is `relative_position_bias_table`, of shape
  `[(2H-1)*(2W-1), heads]`, gathered through a precomputed `[T, T]` index into
  a `[heads, T, T]` bias that is added to the scaled logits before the softmax.
  The bias reaches the kernel as an additive `attn_mask`, so
  `scaled_dot_product_attention` still picks a fused kernel; with
  `causal=True` as well, the causal `-inf` mask and the bias are summed into
  one additive mask rather than handed to the kernel separately. The table is
  constructed at `0.0`, so the layer starts as ordinary attention, and
  construction overrides target it by name
  (`init={"relative_position_bias_table": 0.02}`,
  `trainable={"relative_position_bias_table": False}`). The index is a
  non-persistent `int64` buffer rebuilt from `spatial_shape`, so it stays out
  of checkpoints. `H*W` must equal `T`: a grid that does not cover the
  sequence is `E_CONSTRAINT` at resolution rather than a silent broadcast, and
  `relative_position_bias=True` without a positive `spatial_shape` --- or a
  `spatial_shape` without `relative_position_bias=True` --- is `E_ARGUMENT`.
  The operator carries no windowing of its own: partition a map into windows
  with `chunk`, reassemble it with `concat`, and give the node the `(H, W)` of
  one window.

- **`attention` gains per-projection bias flags.** `qkv_bias` and `out_bias`
  narrow the existing `bias` flag: `q_proj`, `k_proj` and `v_proj` carry a
  bias when `bias and qkv_bias`, and `o_proj` when `bias and out_bias`. Both
  default to `True`, so `bias=` on its own behaves exactly as it did --- one
  flag for all four projections --- while
  `attention(4, qkv_bias=False, out_bias=True)` gives the unbiased
  query/key/value projections with a biased output that ViT-style blocks use.
  A projection built without a bias has no `bias` parameter at all, so its
  `state_dict` is unchanged from `bias=False`.

- **Plan format: `attention` nodes carry four more arguments.** `qkv_bias`,
  `out_bias`, `relative_position_bias` and `spatial_shape` are concrete on
  every resolved `attention` node, so a plan saved by 0.5.0 that contains an
  `attention` node no longer matches its own digest and must be re-resolved
  from its configuration; the configuration itself is unchanged. Plans without
  an `attention` node encode and digest exactly as they did.

- **Faster forward passes.** `GraphModule` no longer re-derives its per-call
  constants on every call. Contract shapes are parsed once at build time and
  resolved against a concrete batch only when the batch size or the runtime
  dtype changes, and each node's module handle, port labels and expected
  shapes are baked into a flat program instead of being rebuilt from f-strings
  and `ModuleDict` lookups per call. The fixed per-node overhead drops from
  about 13 us to about 5 us on the reference machine: in
  `tests/benchmark/test_parity_vs_handwritten_pytorch.py` the MLP's parity
  ratio against hand-written PyTorch improves from 1.41x to 1.19x at batch 256
  and from ~2.5x to ~1.6x at batch 32, and `transformer_block` from ~1.44x to
  ~1.29x. Validation is unchanged: every shape, dtype and device check still
  runs on every port of every call, in the same order, with the same
  `E_RUNTIME` messages.

- **The forward-time state-integrity check replays a recorded walk.** The check
  that catches a module creating or removing registered state mid-forward used
  to rebuild every dotted parameter and buffer name by recursing through the
  module tree on each call. It now records that walk once at build time — the
  modules it visited, each one's child mapping, and the registration keys each
  one contributed — and replays it against that flat list, comparing key
  tuples instead of formatting names. Pinning each visited module's `_modules`
  mapping by key *and* by child identity is what makes the flat list safe:
  nothing can be grafted into the tree unseen, so the recorded visit order and
  every module's path are still the ones the dotted names were built from.
  `None` slots and tensors already seen earlier in the walk are filtered
  exactly as `named_parameters` and `named_buffers` filter them. Detection is
  therefore unchanged from 0.5.0 — including a forward that fills a
  declared-`None` parameter, buffer or submodule slot (the lazy-initialization
  bug, where `build()` hands the caller an empty `parameters()` and the real
  weights appear on the first call, never trained, never seeded, and absent
  from the checkpoint the plan describes), a submodule swapped out under an
  unchanged attribute name, and a parameter dropped with `self.weight = None`
  — except in one case, where it is now *stricter*: attaching, detaching or
  swapping a submodule that registers no parameters or buffers of its own
  contributes no dotted name, so 0.5.0 let that through and this check reports
  it. As in 0.5.0, registering an already-registered tensor under a second
  name is not reported, because de-duplication drops the alias from the old
  name tuple and the new key tuple alike.

## 0.5.0 (2026-09-21)

A minor release adding a first-class `spatial_attention` operator and an
opt-in performance/optimization test suite.

- **Opt-in performance/optimization tests.** tests/benchmark/, marked
  benchmark and excluded from CI (pytest -m "not network and not benchmark"),
  times every network in examples/networks end to end, checks whether
  torch.compile accepts each built graph and by how much it speeds
  inference up, and asserts hndl's built graphs run within a bounded
  multiple of hand-written PyTorch equivalents. Run explicitly with
  pytest -m benchmark tests/benchmark.

- **spatial_attention operator.** The SAGAN self-attention block (Zhang et
  al. 2018) as one first-class operator over [B, C, H, W] feature maps:
  1x1-conv query/key/value projections, a softmax attention map over
  flattened spatial positions, and a zero-initialized learned_scale gate
  on the residual, all in one node instead of eight.
  examples/networks/sagan_attention.hndl now calls spatial_attention()
  instead of composing it by hand.

## 0.4.0 (2026-09-21)

A minor release for HyperGAN's fixed-context discriminator: `concat` joins tensors along the batch axis with the multiple tracked as `k*B`, and a new `chunk` operator cuts an axis back into equal sections.

- **`concat` joins along the batch axis, tracked as `k*B`.** `axis=0` was
  rejected; it now stacks examples, so
  `combined = concat(candidate, context, axis=0)` turns two `[B, C, H, W]`
  tensors into one `[2*B, C, H, W]` and a shared --- often frozen --- network
  sees both branches in a single forward pass. Batch stays symbolic: an
  intermediate contract carries the entry `"2*B"`, meaning two of the plan's
  batches, and the resolver tracks it separately from `"B"` everywhere ---
  through ordinary operators, through `pretrained` (including `layers=`), in
  the saved plan, and in the runtime shape check, which requires `k` times the
  batch of the call. The sum runs both ways, so `2*B + B` is `3*B` and one
  unknown input batch is solved from a known total. External `input_shape` and
  `output_shape` contracts stay one plan batch (`E_SCHEMA`), so a graph chunks
  a joined tensor back before publishing it.

- **New `chunk` operator: one axis into equal sections.** `a, b = chunk(x, 2)`
  halves a feature axis and `a, b = chunk(features, 2, dim=0)` takes a `2*B`
  tensor back apart into two `B` tensors, the inverse of the join above. It
  produces exactly `chunks` outputs (`1 <= chunks <= 32`), returns a tuple and
  clears the current tensor like `split`, and its sections are views, so first
  and second derivatives reach the input through every one of them. Uneven
  divisions fail at resolution with `E_CONSTRAINT` rather than at a runtime
  that would only ever see one batch size: an extent of 5 into 2 sections, a
  `2*B` batch into 3, and a plain `B` batch into 2 are all rejected when the
  plan is resolved. `chunks=1` is allowed and returns a one-tuple, matching
  `pretrained(..., layers=("layer1",))`.

- **Plan format: existing plans are byte-identical.** Batch multiples have one
  spelling each --- `"1*B"`, `"0*B"`, `"B*2"` and leading zeros are rejected
  with `E_SCHEMA` --- and `"B"` keeps its own, so every plan that does not join
  or divide the batch axis encodes and digests exactly as it did in 0.3.0. No
  existing digest changes. A saved plan's internal port contracts may now carry
  `"k*B"` for `2 <= k <= 1024`; external contracts may not.

- **`outputs_from=` accepts an `int` argument.** A variadic output port could
  only take its count from the length of a sequence argument; it now also
  accepts a bounded `int` argument, which is what `chunk`'s `chunks` is.
  Operators declare how they treat axis 0 with `batch="shared"` (the default:
  batch passes through every port) or `batch="relation"`, and a relation can
  read and set batch entries with `s.batch(port)` and `s.share_batch(*ports)`.

## 0.3.0 (2026-09-21)

A minor release for HyperGAN's multiscale discriminators: one pretrained forward pass returns several intermediate layers, operators may declare variadic output ports, and `.hndl` files count as Python on GitHub.

- **Several layers from one pretrained forward pass.** A provider checkpoint
  can now return more than one intermediate tensor:
  `f1, f2, f3 = pretrained("trunk.pth", provider="resnet18", sha256="<64 hex>",
  layers=("layer1", "layer2", "layer3"))` gives the node one output per entry,
  in the order written, each the raw output of that submodule at its native
  shape --- no pooling, no concatenation --- from a single forward pass that
  stops after the last requested submodule, so the tail of the network never
  runs. Every output carries gradients and second derivatives to the network's
  input: each captured tensor is cloned, so an `nn.ReLU(inplace=True)` or a
  residual `+=` later in the pass can neither overwrite the values the hook saw
  nor break backward through them. Both frontends unpack the tuple, which
  clears the current tensor as `split` does, including for a one-entry
  `layers=("layer1",)`; an empty `layers=()` means the argument was not given.
  Duplicate, empty, or unknown entries, a submodule that runs more than once
  before the pass stops, and a conflict with `layer=` or `readout=` each fail
  with `E_PRETRAINED`. Because plan digests cover every canonical argument, a
  plan saved before this release that holds a `pretrained` node fails to
  restore with `E_INTEGRITY` and must be re-resolved once from its source;
  plans without a `pretrained` node are unaffected.
- **Operators may declare variadic output ports.** A shape declaration can
  write `out*` after the arrow together with `outputs_from="<argument>"`, and
  the node expands ordinal ports `out0`, `out1`, ... from that sequence
  argument's length, mirroring the `x0`, `x1` expansion of variadic inputs. An
  empty sequence leaves the single declared `out` port, so such an operator's
  existing plans keep their encoding and digests. `Arg` also gains a `"strs"`
  type: a bounded tuple of strings, written as a tuple literal in
  configuration.
- **`.hndl` files count as Python on GitHub.** A `.gitattributes` at the
  repository root maps `*.hndl` to Python, so Linguist includes network
  definitions in the repository's language statistics.

## 0.2.1 (2026-09-21)

A patch release for HyperGAN's migration: cast networks check inputs in the
dtype they were cast to, and provider checkpoints read tensors through
host-registered readouts.

- **Networks follow `.double()`, `.half()` and `.to(dtype=...)`.** Casting a
  built network casts its parameters, and now its runtime contract as well: a
  network cast to float64 accepts float64 inputs and returns float64 outputs,
  and `.float()` puts it back. Graphs without parameters follow the cast too.
  Integer and boolean ports --- token ids, masks --- keep the dtype the plan
  declared, because those casts leave non-floating state alone. Copies made
  with `copy.deepcopy` carry the dtype the original was cast to, and casting a
  copy leaves the original alone.
- **Named readouts for provider checkpoints.** A `.pth` checkpoint whose useful
  tensor comes from a method rather than `forward` — DINOv2's
  `forward_features(x)["x_norm_patchtokens"]` or
  `get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True)` — now
  names host code for it. `registry.pretrained_provider(name, build,
  readouts={"patch_tokens": fn})` binds named `callable(model, x)` readouts to a
  provider (and `registry.pretrained_readout(provider, name, fn)` adds one
  later), which a network selects with
  `pretrained("w.pth", provider="dino", sha256="<64 hex>",
  readout="patch_tokens")`. Configuration still names only registered code, so
  the plan stays a serializable string. The readout runs on the meta device
  while the plan resolves, so the node's output shape is inferred as usual; it
  must be a pure function of `(model, x)` returning exactly one tensor
  (concatenate or stack inside the readout, or register one readout per
  tensor). `readout=` and `layer=` are mutually exclusive, an unknown name lists
  the readouts the provider offers, and neither applies to transformers or timm
  checkpoints, which select `output=`.

## 0.2.0 (2026-09-21)

A feature release driven by HyperGAN's migration to HNDL: networks copy
with `copy.deepcopy`, graphs take named inputs and publish named outputs,
GAN critics get spectral normalization and deterministic pooling, and
local `.pth` checkpoints load through registered providers.

- **`copy.deepcopy(model)` works.** A built network copies natively, which is
  what an exponential-moving-average model needs. The copy owns every
  parameter and buffer, keeps each `requires_grad` flag and the train/eval
  mode, and draws no random numbers; the resolved plan, its registry and the
  build receipt are immutable, so the copy shares them. `copy.copy(model)` is
  a second handle on the same tensors. Pickling a whole module still cannot
  work --- plans hold frozen mappings and operator declarations hold
  functions --- but it now says so and points at `plan.to_json()` plus
  `torch.save(model.state_dict())` instead of failing inside the pickler.
- **`broadcast_add` operator.** Elementwise sum of two equal-rank tensors with
  numpy-style broadcasting of size-1 non-batch axes, for per-sample,
  per-channel biases such as `[B, C, H, W] + [B, C, 1, 1]`. The backward pass
  reduces the broadcast axes with `sum`, so it is deterministic on CUDA. `add`
  keeps its exact-shape contract.
- **`adaptive_avg_pool` is deterministic and double-differentiable.** When the
  input extents divide the output size it pools by reshaping and taking a
  `mean` over the window axes, and a ragged pool to at most 64 windows writes
  the windows out with slicing and `stack`. Both paths run under
  `torch.use_deterministic_algorithms(True)`, give bit-identical CUDA
  gradients run to run, and support the second derivative a gradient penalty
  needs. Only a ragged pool to more than 64 windows still falls back to
  `F.adaptive_avg_pool2d`, whose CUDA backward is nondeterministic. Values,
  shapes, arguments, and the module `repr` are unchanged.
- **Spectral normalization.** `linear`, `conv`, `conv1d`, and `deconv` accept
  `spectral_norm=True`, wrapping the weight with
  `torch.nn.utils.parametrizations.spectral_norm` for GAN discriminators. The
  learned tensor is then registered as `parametrizations.weight.original`, which
  is the name `init` and `trainable` overrides must target.
- **Local `.pth` checkpoints with custom providers.** `pretrained("weights.pth",
  provider="vgg16", sha256="<hex>", layer="features.16")` loads a plain
  `state_dict` into an architecture the host registered with
  `registry.pretrained_provider(name, build)`. Registration is trusted Python
  on one `Registry` instance; configuration may only name a provider that
  already exists. The digest is required and verified at resolution and again
  before loading, the file is read with `torch.load(..., weights_only=True)`
  and `strict=True`, and `layer=` returns an intermediate submodule's output
  through a forward hook that stops the pass there — what perceptual losses
  and feature matching need.
- **`constant` operator.** Emits a `[B, *shape]` tensor filled with `value`
  (default `0.0`), taking the batch extent and device from the incoming
  tensor and the compute dtype from the plan. It has no parameters and no
  buffers, its output is detached from the autograd graph, and the shape may
  be omitted and read from the output contract.
- **`matmul` operator.** A batched matrix product of two rank-3 tensors,
  `a[B, N, K], b[B, K, M] -> out[B, N, M]`, with both operands explicit. The
  shared inner dimension `K` flows in both directions; other ranks are
  rejected with `E_CONSTRAINT`, so reshape a `[B, C, H, W]` map to
  `[B, C, H*W]` first.
- **`learned_scale` operator.** `out = gamma * x` with one learned scalar
  parameter named `gamma`, started from `init_value` (default `0.0`, the SAGAN
  convention that makes a gated residual begin as the identity). Use `scale`
  for a fixed constant.
- **SAGAN self-attention example.** `examples/networks/sagan_attention.hndl`
  builds the Self-Attention GAN block (Zhang et al. 2018) from registered
  operators alone: 1x1 projections, `reshape`/`transpose`, two `matmul`s
  around a `softmax`, and a `learned_scale` gate on the residual.
- **Named external inputs and outputs.** `input_shape` and `output_shape`
  accept a mapping of named contracts — `{"z": ("B", 128), "y": ("B", 10)}` —
  in every `resolve*`/`network*` entry point, and `input_dtype` accepts one
  dtype per input. Declarative configuration prebinds each input as a local and
  selects each output by binding its name; a callable receives the inputs as
  keyword arguments and returns a mapping of named outputs. A network with one
  input and one output is unchanged, down to the saved plan's bytes.
- **Multi-output forward.** `GraphModule.forward(**inputs)` accepts exactly the
  declared input names and returns a dictionary keyed by the declared output
  names. The `network*` facades accept the inputs positionally or by keyword and
  return a tensor for one output or that dictionary for several.
- **Conditional GAN example.** `examples/networks/conditional_discriminator.hndl`
  takes an image and a label and publishes `logits` and `features`.

## 0.1.2 (2026-09-21)

README links are absolute so they work on the PyPI project page; a test
keeps them that way. No library code changed.

## 0.1.1 (2026-09-21)

A documentation and packaging release. No library code changed.

- **Install from PyPI.** The README now installs with `pip install hndl` and
  links to the documentation site.
- **DCGAN-style generator example.** The transposed-convolution settings are
  spelled out instead of relying on `policy="up2"`.
- **Documentation site.** Published at <https://hypergan.github.io/HNDL/>, built
  with MkDocs and deployed from `master` by `docs.yml`.
- **Release workflow.** PyPI publishing supports either an API token secret or
  Trusted Publishing.

## 0.1.0 (2026-09-21)

The first release of HNDL, the human-readable network definition language.

- **Operator declarations.** Every operation is an `nn.Module` with an
  `@operator(...)` declaration: alias, summary, a shape relation written as a
  small DSL (`x[B, ..., D_in] -> out[B, ..., D_out]`) or a relation function,
  scalar arguments with help text, and runnable examples. Custom operators use
  the same decorator on a `Registry`. Documentation is generated from the
  declarations with `python -m hndl.docs`.
- **Catalog of 59 operators.** Activations (`gelu`, `silu`, `softmax`, …),
  arithmetic and shape plumbing (`add`, `mul`, `transpose`, `permute`, `mean`,
  `pad`, …), normalizations (`layer_norm`, `rms_norm`, `batch_norm`,
  `instance_norm`, `group_norm`, `adaptive_norm`), convolution and spatial
  operators (`conv`, `conv1d`, `deconv`, pooling, `upsample`, `pixel_shuffle`,
  `resblock`, `patch_embed`), sequence operators (`embedding`, `pos_embed`,
  `cls_token`, `attention`, `cross_attention`, `feed_forward`, `swiglu`,
  `transformer_block`, `pool_tokens`, `gather_token`), `moe`, `hopfield`, and
  `dropout`. Convolutions gain the `down2` policy, mirroring `up2`.
- **Sequences and dtypes.** Rank-3 `[B, T, D]` tensors, plan dtypes `float32`,
  `float16`, and `bfloat16`, integer input ports (`input_dtype="int64"`), and
  edge dtype checking.
- **`pretrained(source)`.** Loads any checkpoint that `transformers` or `timm`
  can build from its configuration, from a local directory, a safetensors file,
  or `hf://repo[@revision]`, as one frozen node whose output shape comes from a
  meta-device trace. Optional extra `hndl[pretrained]`.
- **Authored networks.** Complete configs for LeNet-5, DCGAN, U-Net, an
  autoencoder, ResNet-18, a tiny ViT and GPT, a Hopfield classifier, and a
  mixture-of-experts transformer under `examples/networks`, tested and rendered
  into `docs/networks.md`.
- **Backend.** PyTorch is a dependency. Registered storage is bounded by an
  allocation-free meta-device pass before construction; the build receipt
  records the dtype and state bytes.
- **Removed.** `Registry.register`, `register_torch`, `ShapeRule`, `Dim`,
  `preserves_shape`, and the per-node `state_bytes`/`state_version` plan
  fields from the 0.1.0a1 alpha.
