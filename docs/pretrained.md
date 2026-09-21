# Pretrained networks

`pretrained(source)` loads a checkpoint as one node of an HNDL network. The
checkpoint's own configuration defines the architecture: HNDL does not
describe GPT-2 or ResNet layouts. `transformers` builds anything with a
`model_type`, `timm` builds anything with an `architecture`, and HNDL wraps
the result, infers its output shape, freezes it, and lets you keep building.

```python
from hndl.torch import network

model = network(
    """
    pretrained("hf://openai-community/gpt2", output="features", name="gpt2")
    linear(2)
    """,
    input_shape=("B", 128),
    input_dtype="int64",
    output_shape=("B", 128, 2),
    dtype="bfloat16",
    device="cuda:0",
)
```

Install the optional dependencies with `pip install 'hndl[pretrained]'`
(safetensors, huggingface_hub, transformers, timm).

## Sources

| `source` | Meaning |
| --- | --- |
| `hf://owner/repo` | A Hugging Face Hub repository at its default branch |
| `hf://owner/repo@revision` | A branch, tag, or commit of that repository |
| `/path/to/checkpoint` | A directory holding `config.json` and safetensors weights (the layout `save_pretrained` writes) |
| `/path/to/model.safetensors` with `config="/path/to/config.json"` | A bare weights file plus its configuration |

A `.safetensors` file alone is not enough for any loader: tensors do not
describe an architecture. Nothing in a checkpoint is executed as code.

Resolution reads only `config.json` (downloading it for `hf://` sources when
it is not cached) and records the resolved commit hash or content digest as
the node's `revision` argument. Weights are downloaded and loaded when the
network is built. Restoring a saved plan re-resolves the source; if it now
points at a different checkpoint the restore fails with `E_CONSTRAINT`.

## Inputs and outputs

The input contract follows the checkpoint:

- Text models consume token ids. Declare the graph input as `("B", T)` with
  `input_dtype="int64"`.
- Vision models consume images `("B", C, H, W)` at the checkpoint's native
  resolution; the resolver fixes `C`, `H`, and `W`, and a different size fails
  with `E_CONSTRAINT`.

`output` selects the tensor the node returns:

| `output` | transformers | timm |
| --- | --- | --- |
| `"features"` (default) | `last_hidden_state`, `[B, T, D]` or `[B, N, D]` | `forward_features`, e.g. `[B, C, h, w]` |
| `"pooled"` | `pooler_output`, `[B, D]` | `forward_head(pre_logits=True)`, `[B, D]` |
| `"logits"` | The `*ForCausalLM` / `*ForImageClassification` / `*ForSequenceClassification` head | The classifier, `[B, classes]` |
| `"embeds"` | Projected embeddings of a CLIP-style tower (`image_embeds` / `text_embeds`) | — |
| any attribute name | That field of the model output | — |

Multi-modal checkpoints such as CLIP have two towers; pick one with
`component="vision"` or `component="text"`.

The output shape is discovered by running the architecture on PyTorch's
`meta` device with the declared input shape, so no weights are read during
resolution and `print(plan)` shows the real shapes.

## Freezing and dtype

The wrapped model is frozen (`requires_grad=False`) and stays in eval mode
even when the surrounding network is in training mode, so dropout and batch
statistics of the checkpoint do not move while you train new layers. Pass
`trainable=True` to fine-tune it; `model.train()` then reaches the checkpoint
as well.

Weights are loaded in float32 and cast to the plan `dtype` (`float32`,
`float16`, or `bfloat16`). Reduced precision is qualified on CUDA.

## Tested checkpoints

The `network`-marked tests exercise, against the reference libraries:

| Checkpoint | Call | Output |
| --- | --- | --- |
| GPT-2 | `pretrained("hf://openai-community/gpt2", output="logits")` | `[B, T, 50257]` |
| ResNet-18 (timm) | `pretrained("hf://timm/resnet18.a1_in1k", output="logits")` | `[B, 1000]` |
| ViT-B/16 | `pretrained("hf://google/vit-base-patch16-224", output="logits")` | `[B, 1000]` |
| CLIP ViT-B/32 | `pretrained("hf://openai/clip-vit-base-patch32", component="vision", output="embeds")` | `[B, 512]` |
| DINOv2 small | `pretrained("hf://facebook/dinov2-small", output="pooled")` | `[B, 384]` |

Any other checkpoint that transformers or timm can build from its
configuration works the same way.
