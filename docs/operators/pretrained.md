# `pretrained`

Load a pretrained network from disk or the Hugging Face Hub as one frozen node.

**Category:** pretrained · **Identity:** `pretrained@1`

## Shape

```text
x:any -> out
```

Relation: `input contract and output shape come from the checkpoint's configuration (meta-device trace)`

| Port | Direction | Pattern | dtype |
| --- | --- | --- | --- |
| `x` | input | `x` | any |
| `out` | output | `out` | compute |

## Arguments

| Name | Type | Default | Constraints | Description |
| --- | --- | --- | --- | --- |
| `source` (positional) | str | required | — | Checkpoint location: a directory with config.json and safetensors weights, a .safetensors file (with config=), hf://owner/repo[@revision], or a local .pth state dict (with provider= and sha256=). |
| `output` | str | `"features"` | — | Which tensor to return: "features" (last hidden state), "pooled", "logits", "embeds" (projected CLIP-style embeddings), or a raw output attribute name. |
| `component` | str | `""` | — | Tower of a multi-modal checkpoint such as CLIP: "vision" or "text". |
| `config` | str | `""` | — | Path to config.json when source is a bare .safetensors file. |
| `provider` | str | `""` | — | Name of an architecture builder the host registered with registry.pretrained_provider(name, build); required for a local .pth state dict. |
| `sha256` | str | `""` | — | The 64 hex character digest of a local .pth file, verified before it is loaded. |
| `layer` | str | `""` | — | Dotted named_modules() path of the provider submodule whose output the node returns, such as "features.16"; empty returns the model's own output. |
| `readout` | str | `""` | — | Name of a readout the host registered with the provider, such as "patch_tokens"; the node returns that callable's tensor instead of the model's own output. Mutually exclusive with layer=. |
| `revision` | str | inferred | — | Resolved commit hash or content digest. Filled in at resolution and checked on restore. |

## Description

Wraps a checkpoint whose architecture is defined by its own configuration.

The source's ``config.json`` selects the implementation: a ``model_type``
is built by ``transformers`` (``AutoModel``, the ``*ForCausalLM`` /
``*ForImageClassification`` heads for ``output="logits"``, or the
``<Model>VisionModel`` / ``<Model>TextModel`` tower named by
``component``), and a timm ``architecture`` is built by ``timm``. HNDL
does not describe these layouts; it traces the model on PyTorch's meta
device to learn the output shape, then loads the weights at build time.

Text models consume token ids: declare the graph input as ``[B, T]`` with
``input_dtype="int64"``. Vision models consume ``[B, C, H, W]`` at the
checkpoint's native resolution, which the relation fixes. The wrapped
model is frozen and kept in eval mode by default; pass ``trainable=True``
to fine-tune it, which also lets ``model.train()`` reach it. Weights are
loaded in float32 and cast to the plan dtype. Requires the optional
``hndl[pretrained]`` extra.

The plan records the resolved ``revision`` so a restored plan fails
(``E_CONSTRAINT``) if the source now points at a different checkpoint.

A local ``.pth`` state dict describes no architecture, so it names one the
host registered as trusted Python:
``registry.pretrained_provider("vgg16", build)`` binds a zero-argument
callable returning the ``nn.Module``, and configuration may only name an
already registered provider. Such a source is written
``pretrained("/path/weights.pth", provider="vgg16", sha256="<64 hex>",
layer="features.16")``. The ``sha256`` is required, is verified against the
file at resolution and again before loading, and the weights are read with
``torch.load(..., weights_only=True)``, which unpickles no objects; keys
must match exactly (``strict=True``). ``layer`` names a submodule by its
dotted ``named_modules()`` path and returns that submodule's output through
a forward hook, stopping the pass there; omitted, the node returns the
model's own output. Provider checkpoints declare no input contract, so the
graph input shape is whatever the module accepts (floating point); the
meta-device trace checks it.

``readout=`` names host code instead of a submodule, for checkpoints whose
useful tensor comes from a method rather than ``forward``. The host binds
named ``callable(model, x)`` readouts to the provider, and configuration may
only name one of them::

    registry.pretrained_provider("dinov2_vits14", build_dinov2, readouts={
        "patch_tokens": lambda m, x: m.forward_features(x)["x_norm_patchtokens"],
        "layers_2_5_8_11": lambda m, x: torch.cat(
            m.get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True), dim=1),
    })

which a network then selects with
``pretrained("/path/dinov2_vits14.pth", provider="dinov2_vits14",
sha256="<64 hex>", readout="patch_tokens")``. Readouts can also be added to
an existing provider with
``registry.pretrained_readout("dinov2_vits14", "cls_token", fn)``. A readout
must return exactly one tensor --- concatenate or stack several inside the
readout, or register one readout per tensor --- and it runs during
resolution on PyTorch's meta device, so it must be a pure function of
``(model, x)`` that touches no real data. ``readout=`` and ``layer=`` are
mutually exclusive, an unknown readout name fails with ``E_PRETRAINED``
listing the ones registered for that provider, and neither applies to
transformers or timm checkpoints, which select ``output=``.

## Examples

### Example 1

GPT-2 hidden states feeding a new trainable head.

```python
pretrained("hf://openai-community/gpt2", output="features", name="gpt2")
linear(2)
```

Input `['B', 16]` (`input_dtype="int64"`) → output `['B', 16, 2]`.

This example downloads a checkpoint; resolved shapes and parameter counts depend on it.

### Example 2

ImageNet logits from a timm ResNet-18.

```python
pretrained("hf://timm/resnet18.a1_in1k", output="logits")
```

Input `['B', 3, 224, 224]` → output `['B', 1000]`.

This example downloads a checkpoint; resolved shapes and parameter counts depend on it.

### Example 3

The projected image embedding of CLIP's vision tower.

```python
pretrained("hf://openai/clip-vit-base-patch32", component="vision", output="embeds")
```

Input `['B', 3, 224, 224]` → output `['B', 512]`.

This example downloads a checkpoint; resolved shapes and parameter counts depend on it.
