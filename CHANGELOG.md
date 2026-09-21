# Changelog

## Unreleased

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
