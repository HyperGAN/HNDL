# Changelog

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
