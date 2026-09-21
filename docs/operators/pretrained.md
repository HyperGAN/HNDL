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
| `source` (positional) | str | required | — | Checkpoint location: a directory with config.json and safetensors weights, a .safetensors file (with config=), or hf://owner/repo[@revision]. |
| `output` | str | `"features"` | — | Which tensor to return: "features" (last hidden state), "pooled", "logits", "embeds" (projected CLIP-style embeddings), or a raw output attribute name. |
| `component` | str | `""` | — | Tower of a multi-modal checkpoint such as CLIP: "vision" or "text". |
| `config` | str | `""` | — | Path to config.json when source is a bare .safetensors file. |
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
