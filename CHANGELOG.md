# Changelog

## Unreleased

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
