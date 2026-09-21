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
| `/path/to/weights.pth` with `provider="<name>"` and `sha256="<hex>"` | A local state dict loaded into an architecture the host registered (see below) |

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

## Local checkpoints and custom providers

A `.pth` file is a plain `state_dict`: it carries tensors, not a layout, and
no `config.json` tells HNDL what to build. Point it at an architecture the
host registered on its `Registry` as ordinary trusted Python. Registration is
never configuration: a config may only *name* a provider that already exists,
and providers belong to the registry they were added to, so one
`Registry.builtins()` never sees another's.

```python
import hashlib

import torch
from torch import nn

from hndl import Registry
from hndl.torch import network


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


# Whatever wrote the checkpoint, here a plain state_dict on disk.
torch.save(Encoder().state_dict(), "encoder.pth")
digest = hashlib.sha256(open("encoder.pth", "rb").read()).hexdigest()

registry = Registry.builtins()
registry.pretrained_provider("encoder", Encoder)   # trusted host code

model = network(
    f"""
    pretrained("encoder.pth", provider="encoder", sha256="{digest}", layer="features.2", name="perceptual")
    conv(8, kernel_size=3, padding=1)
    """,
    input_shape=("B", 3, 32, 32),
    output_shape=("B", 8, 32, 32),
    registry=registry,
    device="cuda:0",
)
```

Compute the digest with `python -c 'import hashlib,sys;
print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' encoder.pth`,
or read it from the `E_PRETRAINED` message raised when `sha256=` is missing.

- **`provider=`** names a zero-argument callable registered with
  `registry.pretrained_provider(name, build)` that returns an `nn.Module`
  with its architecture already defined. HNDL loads the checkpoint into it;
  the builder itself must not read the file.
- **`sha256=`** is required for a `.pth` source. It is verified when the plan
  resolves and again immediately before the weights are read, so a file that
  changed under a saved plan fails with `E_PRETRAINED` instead of loading
  different weights. The plan also records the digest as its `revision`.
- The file is read with `torch.load(path, map_location="cpu",
  weights_only=True)` — tensors and plain containers only, never pickled
  objects — and loaded with `strict=True`; missing or unexpected keys fail
  with `E_PRETRAINED`.
- **`layer=`** names a submodule by its dotted `named_modules()` path, such as
  `"features.2"` or `"layer3.1.conv2"`. The node runs the model with a forward
  hook on that submodule, returns its output, and stops the pass there, which
  is what perceptual losses and feature matching want. An unknown name fails
  with `E_PRETRAINED` listing the available submodules. Omit `layer` and
  `readout` and the node returns the model's own return value. `output=`,
  `component=` and `config=` apply to transformers and timm checkpoints, not to
  providers.
- **`readout=`** names host code instead of a submodule, for checkpoints whose
  useful tensor comes from a method rather than `forward` — DINOv2's
  `forward_features(x)["x_norm_patchtokens"]`, for example. The host binds named
  `callable(model, x)` readouts to the provider and configuration may only name
  one of them:

  ```python
  registry.pretrained_provider("dinov2_vits14", build_dinov2, readouts={
      "patch_tokens": lambda m, x: m.forward_features(x)["x_norm_patchtokens"],
      "layers_2_5_8_11": lambda m, x: torch.cat(
          m.get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True), dim=1),
  })
  ```

  ```python
  pretrained("dinov2_vits14.pth", provider="dinov2_vits14", sha256="<64 hex>",
             readout="patch_tokens")
  ```

  `registry.pretrained_readout("dinov2_vits14", "cls_token", fn)` adds one to a
  provider that is already registered, and
  `registry.pretrained_readouts("dinov2_vits14")` lists what a provider offers.
  A readout must return exactly one tensor: combine several with `torch.cat` or
  `torch.stack` inside the readout, or register one readout per tensor you need,
  or anything else fails with `E_PRETRAINED`. It also runs on the `meta` device
  while the plan resolves, so it must be a pure function of `(model, x)` that
  touches no real data and allocates nothing outside the model. An unregistered
  name fails with `E_PRETRAINED` listing that provider's readouts, and
  `readout=` and `layer=` are mutually exclusive — they both say what the node
  returns.
- The whole module is still constructed, so the layers after `layer=` are
  registered (and counted by `parameter_counts`) even though they never run.
  Return a truncated module from the builder if you want them gone.
- The input contract is whatever the module accepts: HNDL fixes no channel
  count or resolution, it only requires a floating-point input and traces the
  architecture on the `meta` device with your declared shape to learn the
  output shape. Resolution reads the file only to hash it; no weights are
  loaded until the network is built, and `parameter_counts(plan)` does not
  touch the file at all.

## Freezing and dtype

The wrapped model is frozen (`requires_grad=False`) and stays in eval mode
even when the surrounding network is in training mode, so dropout and batch
statistics of the checkpoint do not move while you train new layers. Pass
`trainable=True` to fine-tune it; `model.train()` then reaches the checkpoint
as well. Provider checkpoints behave the same way: the node freezes every
parameter it loaded, `model.train()` leaves it in eval mode, and gradients
still flow through it to earlier trainable layers.

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
