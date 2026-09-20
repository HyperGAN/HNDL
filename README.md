# HNDL

**Human-readable Network Definition Language**, pronounced “handle.”

Write your network as a few lines of text. Give HNDL its input and output shapes, and it works out the connecting dimensions. Inspect what it built, access individual layers, and use the model in your PyTorch training code.

**Status: design stage.** The API and console output below describe the proposed behavior. There is no implementation yet.

## A network in a string

Each line starts with an operation. Leave out a dimension to let HNDL infer it, and optionally add `name=` for named access. Input and output constraints belong in the API call:

```python
from hndl.torch import network

model = network(
    """
    linear 64 name=hidden
    relu name=activation
    linear name=scores
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
```

Here, `B` is a variable batch size. The final `linear` has no width argument, so the output constraint determines its 10 output features. Names are optional: `linear 64`, `relu`, and `linear` describe the same operations without custom names. HNDL resolves the dimensions before constructing any modules.

Use `print(model)` to inspect it. HNDL supplies the shape table as its module representation:

```pycon
>>> print(model)
Network: [B, 128] -> [B, 10]  dtype=float32
index  name        operation  input shape  output shape
0      hidden      linear     [B, 128]     [B, 64]
1      activation  relu       [B, 64]      [B, 64]
2      scores      linear     [B, 64]      [B, 10]

>>> model[0]
Linear(in_features=128, out_features=64, bias=True)
>>> model["scores"]
Linear(in_features=64, out_features=10, bias=True)
```

The result is an ordinary `torch.nn.Module` with sequential indexing. Calling it returns a tensor:

```python
import torch

x = torch.randn(8, 128)
scores = model(x)                        # Tensor with shape [8, 10]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

first_layer = model[0]                   # The actual registered module
last_layer = model[-1]
features = model[:2]                     # nn.Sequential sharing these layers
```

Named and integer access return the same modules used by the network. Slices reuse their parameters, so training a slice also updates the original model. The complete model retains its resolved shape contract; a slice is a regular PyTorch sequence. Standard `state_dict()`, `train()`, and `eval()` remain available.

Initial weights use PyTorch’s normal random state. For repeatable initialization in the same environment, call [`torch.manual_seed(7)`](https://docs.pytorch.org/docs/stable/notes/randomness.html#pytorch-random-number-generator) before constructing the network.

## Change the target, keep the definition

A generator starts with 128 features and produces a 32 × 32 RGB image. Choose the channels and three upsampling stages; let the target determine the projection width and starting height and width.

```python
generator_dsl = """
linear name=project
relu name=project_relu
reshape 512 name=seed
deconv 256 policy=up2 name=up1
relu name=act1
deconv 128 policy=up2 name=up2
relu name=act2
deconv 64 policy=up2 name=up3
relu name=act3
conv 3 kernel_size=3 stride=1 padding=1 name=rgb
tanh name=range
"""

generator = network(
    generator_dsl,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
```

The `up2` policy selects transposed-convolution settings that exactly double height and width. `reshape 512` fixes the channels; the following convolution requires an image-shaped tensor, so HNDL infers the remaining height and width. Shapes include batch, channels, height, and width:

```pycon
>>> print(generator)
Network: [B, 128] -> [B, 3, 32, 32]  dtype=float32
index  name          operation  input shape       output shape
0      project       linear     [B, 128]          [B, 8192]
1      project_relu  relu       [B, 8192]         [B, 8192]
2      seed          reshape    [B, 8192]         [B, 512, 4, 4]
3      up1           deconv     [B, 512, 4, 4]    [B, 256, 8, 8]
4      act1          relu       [B, 256, 8, 8]    [B, 256, 8, 8]
5      up2           deconv     [B, 256, 8, 8]    [B, 128, 16, 16]
6      act2          relu       [B, 128, 16, 16]  [B, 128, 16, 16]
7      up3           deconv     [B, 128, 16, 16]  [B, 64, 32, 32]
8      act3          relu       [B, 64, 32, 32]   [B, 64, 32, 32]
9      rgb           conv       [B, 64, 32, 32]   [B, 3, 32, 32]
10     range         tanh       [B, 3, 32, 32]    [B, 3, 32, 32]
```

Three doublings require a 4 × 4 seed, so the projection produces `512 × 4 × 4 = 8192` values. Change only `output_shape` and construct a new model:

| Output constraint | Resolved seed | Projection width |
| --- | --- | --- |
| `("B", 3, 32, 32)` | `[B, 512, 4, 4]` | 8192 |
| `("B", 3, 64, 64)` | `[B, 512, 8, 8]` | 32768 |
| `("B", 3, 32, 64)` | `[B, 512, 4, 8]` | 16384 |

With a 30 × 30 target, resolution fails before allocating model parameters:

```text
E_CONSTRAINT: output height 30 conflicts with up1 -> up2 -> up3.
Three doubling stages require an integer seed height; 30 / 8 = 3.75.
Choose a target divisible by 8 or explicitly change the architecture.
```

Literal values remain constraints. Writing `linear 128 name=project` would also fail for the 32 × 32 target because the seed needs 8192 values. Omitted dimensions are inferred when the constraints determine them; a named policy supplies declared construction choices. Multiple valid choices are reported as ambiguous when no selected policy chooses among them. HNDL does not silently crop, broadcast, or replace layers to make them fit.

## Register your own operation

Extend the vocabulary with a PyTorch module and its shape rule. Here is a proposed registration for the shape-preserving SiLU activation:

```python
from torch import nn
from hndl import Registry, preserves_shape
from hndl.torch import network, register_torch

registry = Registry.builtins()
registry.register(
    "silu",
    identity="example.silu",
    version=1,
    shape=preserves_shape,
)
register_torch(registry, "silu", module=nn.SiLU, state_version=1)

model = network(
    """
    linear 64 name=hidden
    silu name=activation
    linear name=scores
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    registry=registry,
    device="cpu",
)
```

The DSL now understands `silu`. `preserves_shape` tells the resolver that input and output dimensions, layout, and dtype are equal, so constraints propagate in both directions. The backend constructs an `nn.SiLU` for execution. Registration carries the operation's version; network text uses its plain name.

This helper covers unary operations with no author arguments. Operations that change shapes or accept several inputs need their own rules and port declarations, described in [SPEC.md](SPEC.md#8-custom-operators-and-minimal-graphs). Custom implementations still need numerical and gradient checks; declaring a shape rule does not prove their code correct.

## Experiment with less boilerplate

Change `linear 64` to `linear 128`, resolve the same input/output constraints, and send the resulting model through your existing training and evaluation loop. Compare accuracy, loss, or inference time while keeping architecture definitions small and readable. HNDL handles the connecting dimensions; your application owns metrics, optimizers, and the search over candidates.

For inspection without building a model, use the pure resolver:

```python
from hndl import resolve

plan = resolve(
    generator_dsl,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
)
print(plan)
```

Built-in resolution needs no PyTorch import or tensor allocation. Save the resolved plan with your experiment to record exactly which architecture was constructed. Named graphs and custom multi-input operations are also part of the proposed v1; the text DSL starts with sequences.

[SPEC.md](SPEC.md) defines the language, registration, shape rules, and PyTorch interface. [DESIGN.md](DESIGN.md) preserves the original rationale; the spec reflects the current DSL-focused API.
