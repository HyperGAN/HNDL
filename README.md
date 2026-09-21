# HNDL

**Human-readable Network Definition Language**, pronounced “handle.”

Write your network in Python syntax, as a declarative config or a Python function. Give HNDL its input and output shapes, and it works out the connecting dimensions. Inspect what it built, access individual layers, and use the model in your PyTorch training code.

**Status: initial alpha (`0.1.0a1`).** Config strings, native Python functions, shape inference, branches, and the PyTorch backend are implemented. The [implementation notes](IMPLEMENTATION.md) distinguish this release from the remaining v1 work in the spec. Display spacing and diagnostics below are illustrative.

From a checkout, install on Linux with Python 3.11–3.14:

```sh
python -m pip install -e .
```

PyTorch is a dependency. CPU and CUDA devices are supported; select the device explicitly. Wheel and source distributions are built by CI. PyPI publication requires the one-time setup described in [CONTRIBUTING.md](CONTRIBUTING.md#publishing).

## A network in a string

Write the operations in order. Each takes the current tensor by default and makes its result current; the final result is the network output. Leave out a dimension to let HNDL infer it. Input and output constraints belong in the API call:

```python
from hndl.torch import network

model = network(
    """
    linear(64)
    relu()
    linear()
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
```

Here, `B` is a variable batch size. The final `linear()` has no width argument, so the output constraint determines its 10 output features. HNDL resolves the dimensions before constructing any modules. The implicit current tensor starts at the network input, so a simple chain needs no assignments or tensor arguments.

Activations are explicit operations. Here, `relu()` follows the first linear layer; the final linear layer has no activation. Place an activation wherever you want it in the sequence. `linear`, `conv`, and `deconv` do not add one automatically.

This string is a declarative subset of Python: registered operation calls, optional assignments, literal arguments, and comments. HNDL parses it into a graph without executing it as Python.

Use `print(model)` to inspect it. HNDL supplies the shape table as its module representation:

```pycon
>>> print(model)
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shape  output shape
0      n0    linear     [B, 128]     [B, 64]
1      n1    relu       [B, 64]      [B, 64]
2      n2    linear     [B, 64]      [B, 10]

>>> model[0]
Linear(in_features=128, out_features=64, bias=True)
>>> model[-1]
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

Unnamed operations receive IDs such as `n0`; `model["n0"]` and `model[0]` return the same module. Optional names appear in the branching example below. Slices reuse their parameters, so training a slice also updates the original model. The complete model retains its resolved shape contract; a slice is a regular PyTorch sequence. Standard `state_dict()`, `train()`, and `eval()` remain available.

Initial weights use PyTorch’s normal random state. For repeatable initialization in the same environment, call [`torch.manual_seed(7)`](https://docs.pytorch.org/docs/stable/notes/randomness.html#pytorch-random-number-generator) before constructing the network.

## The same network in native Python

Use the same implicit current tensor inside an ordinary Python function:

```python
from hndl import ops
from hndl.torch import network_from_callable


def classifier(x):
    ops.linear(64)
    ops.relu()
    ops.linear()


model = network_from_callable(
    classifier,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
```

This builds the same network and shape table as the string example. HNDL calls `classifier` once with a symbolic input, which also becomes current. Falling through without a return selects the final current tensor. Use `x` when you need an explicit reference to the original input; simple chains can leave it unused.

Python helpers and loops run during construction. The resulting model runs the captured operations on real tensors during `model(x)`; it does not rerun your authoring function. Tensor-dependent Python branching is unsupported. Native authoring runs as ordinary trusted Python with your process's permissions; config strings use the restricted loader described below. Both frontends share operations, shape rules, and the PyTorch backend.

## Change the target, keep the definition

A generator starts with 128 features and produces a 32 × 32 RGB image. Choose the channels and three upsampling stages; let the target determine the projection width and starting height and width.

```python
generator_config = """
linear()
relu()
reshape(512)
deconv(256, policy="up2")
relu()
deconv(128, policy="up2")
relu()
deconv(64, policy="up2")
relu()
conv(3, kernel_size=3, stride=1, padding=1)
tanh()
"""

generator = network(
    generator_config,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
```

The `up2` policy selects transposed-convolution settings that exactly double height and width. `reshape(512)` fixes the channels; the following convolution requires an image-shaped tensor, so HNDL infers the remaining height and width. Shapes include batch, channels, height, and width:

```pycon
>>> print(generator)
Network: [B, 128] -> [B, 3, 32, 32]  dtype=float32
index  name  operation  input shape       output shape
0      n0    linear     [B, 128]          [B, 8192]
1      n1    relu       [B, 8192]         [B, 8192]
2      n2    reshape    [B, 8192]         [B, 512, 4, 4]
3      n3    deconv     [B, 512, 4, 4]    [B, 256, 8, 8]
4      n4    relu       [B, 256, 8, 8]    [B, 256, 8, 8]
5      n5    deconv     [B, 256, 8, 8]    [B, 128, 16, 16]
6      n6    relu       [B, 128, 16, 16]  [B, 128, 16, 16]
7      n7    deconv     [B, 128, 16, 16]  [B, 64, 32, 32]
8      n8    relu       [B, 64, 32, 32]   [B, 64, 32, 32]
9      n9    conv       [B, 64, 32, 32]   [B, 3, 32, 32]
10     n10   tanh       [B, 3, 32, 32]    [B, 3, 32, 32]
```

Three doublings require a 4 × 4 seed, so the projection produces `512 × 4 × 4 = 8192` values. Change only `output_shape` and construct a new model:

| Output constraint | Resolved seed | Projection width |
| --- | --- | --- |
| `("B", 3, 32, 32)` | `[B, 512, 4, 4]` | 8192 |
| `("B", 3, 64, 64)` | `[B, 512, 8, 8]` | 32768 |
| `("B", 3, 32, 64)` | `[B, 512, 4, 8]` | 16384 |

With a 30 × 30 target, resolution fails before allocating model parameters:

```text
E_CONSTRAINT: output height 30 conflicts with the three deconv stages (n3, n5, n7).
Three doubling stages require an integer seed height; 30 / 8 = 3.75.
Choose a target divisible by 8 or explicitly change the architecture.
```

Literal values remain constraints. Writing `linear(128)` would also fail for the 32 × 32 target because the seed needs 8192 values. Omitted dimensions are inferred when the constraints determine them; a named policy supplies declared construction choices. Multiple valid choices are reported as ambiguous when no selected policy chooses among them. HNDL does not silently crop, broadcast, or replace layers to make them fit.

## Split and reuse tensors

Tuple assignment names the two results of a split. Pass either result into later operations, and use `#` for comments:

```python
branches = network(
    """
    # Take 64 features, then keep the remainder. dim defaults to 1.
    z1, z2 = split(64, name="partition")  # Both are [B, 64] here.

    linear(z1, 128, name="content")
    features = relu(name="features")
    style = linear(z2, 128, name="style")
    add(features, style, name="combined")
    """,
    input_shape=("B", 128),
    output_shape=("B", 128),
    device="cpu",
)
```

`split(64)` takes the current tensor and produces exactly two outputs: the first 64 entries along `dim`, and everything after them. With 160 input features, the outputs would have 64 and 96 features. Use `dim=2`, for example, to split an image's height. Both pieces must be nonempty, and splitting the batch dimension is excluded.

After a split there are two results, so there is no single current tensor. `linear(z1, 128)` explicitly starts the content branch; `relu()` then follows it implicitly. Passing `z2` starts the style branch. Joins such as `add(features, style)` take both inputs explicitly.

The graph display includes every input and output port:

```pycon
>>> print(branches)
Network: [B, 128] -> [B, 128]  dtype=float32
name       operation  input shapes                                    output shapes
partition  split      x=input:[B, 128]                                first=[B, 64], rest=[B, 64]
content    linear     x=partition.first:[B, 64]                        out=[B, 128]
features   relu       x=content.out:[B, 128]                           out=[B, 128]
style      linear     x=partition.rest:[B, 64]                         out=[B, 128]
combined   add        a=features.out:[B, 128], b=style.out:[B, 128]     out=[B, 128]
```

`z1` and `z2` are tensor references inside the definition. `name="partition"` names the split layer, accessible as `branches["partition"]`. Each operation runs once per forward call; both branches share the split results and retain normal gradient flow.

This branched model still accepts and returns a tensor. Layer lookup uses names; positional indexing and slicing apply to simple chains. The final current tensor is the output by default; ending on a split requires an explicit selection. An optional `out` binding overrides that choice: a split-only config can finish with `out = z2` to return the remainder. Assignments store references; they do not change the current tensor themselves.

### Feed a branch into adaptive normalization

The built-in `adaptive_norm` operation takes two inputs: image features and per-channel style parameters. A config can route a split directly into it, as the runnable [adaptive-normalization example](examples/adaptive_normalization.py) does:

```python
z1, z2 = split(64)
linear(z1)                             # Projection resolves to 512.
features = reshape(32, 4, 4)
adaptive_norm(features, z2)             # 32 channels need 64 style parameters.
```

Use input shape `("B", 128)` and output shape `("B", 32, 4, 4)`. This operation expects one scale adjustment and one bias per channel, so `z2` already has the required width. To learn a style mapping first, replace the final line with:

```python
linear(z2, 128)
relu()
style = linear()                       # Final width resolves to 64.
adaptive_norm(features, style)
```

Explicit inputs and nested calls also work, for example `linear(relu(linear(z2, 128)))`. A reusable subnetwork can also be registered as an operation with declared ports and shape rules, then called as `style = style_mlp(z2)`. Each call creates its own module instance when the model is built.

### Reuse a Python helper

Native helpers share the current tensor of their enclosing definition. Pass an explicit input when starting a branch, then let consecutive calls use the current tensor:

```python
def mlp(first, width):
    ops.linear(first, width)
    ops.relu()
    return ops.linear(width)


def architecture(x):
    z1, z2 = ops.split(64)
    content = mlp(z1, 128)
    style = mlp(z2, 128)
    ops.add(content, style)


branches = network_from_callable(
    architecture,
    input_shape=("B", 128),
    output_shape=("B", 128),
    device="cpu",
)
```

The helper explicitly starts each branch from its argument and returns a reference for the join. Native functions can also return a tensor explicitly to select the network output. Variable names do not name layers; optional `name="..."` arguments give layers stable identities for lookup and saved state.

## Set initial values and freeze layers

Keep initialization choices with the network definition:

```python
model = network(
    """
    linear(64, trainable=False)
    relu()
    linear(init={"weight": 0, "bias": 0})
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
```

The first layer keeps its normal initial values and has frozen parameters. The final layer starts with zero weights and bias and remains trainable. This illustrates the controls; choose initial values to suit your architecture. The [adaptive-normalization example](examples/adaptive_normalization.py) uses a zero-initialized style projection to start with unit scale and zero bias on normalized features.

`init` maps parameter names to constant values. `trainable=False` freezes every parameter in that operation; use `trainable={"weight": False}` to freeze only its weight. Unspecified parameters retain their module defaults. Custom operations can use exact nested parameter names such as `"projection.weight"`. These options work in native Python too, for example `ops.linear(64, trainable=False)`.

Both settings are saved in the resolved plan and applied whenever you build it. They do not change the inferred shapes. Gradients still flow through frozen layers to their inputs, and `model.train()` and `model.eval()` work normally. A state dictionary saves parameter values, so keep the plan alongside it to reconstruct the architecture and trainability settings.

## Load a config file

Save declarative operation calls and optional assignments in a file such as `generator.hndl`, using the same syntax as `generator_config` above. Load it with constraints from your application:

```python
from hndl.torch import network_file

generator = network_file(
    "generator.hndl",
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
```

File loading reads bounded UTF-8 text and uses the same declarative parser as `network(...)`. There are no imports, attribute lookups, loops, or arbitrary function calls in configs. Calls identify operations already registered by your application. A config cannot register or import an implementation.

The loader translates an explicitly allowed subset of Python's AST into graph data. It never executes config code with `eval` or `exec`, and invalid input never falls back to native Python. The loader applies source, parser, graph, and model-size limits and parses all declarative input in an isolated worker. Even AST parsing can exhaust resources, so syntax restrictions alone are insufficient. See the [loading and trust contract](SPEC.md#loading-limits-and-trust-boundaries). Registered implementations remain trusted application code.

## Register your own operation

Every operation, built-in or yours, is an `nn.Module` with an `@operator` declaration above it. The declaration names the operation, describes its shape relation, and documents its arguments; the class is the implementation:

```python
from torch import nn
from hndl import Registry
from hndl.torch import network

registry = Registry.builtins()

@registry.operator(
    "my_silu",
    identity="example.silu",
    summary="Sigmoid-weighted linear unit.",
    shape="x[B, ...] -> out[B, ...]",
)
class SiLU(nn.SiLU):
    """Computes ``x * sigmoid(x)`` elementwise."""

model = network(
    """
    linear(64)
    my_silu()
    linear()
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    registry=registry,
    device="cpu",
)
```

Configs now understand `my_silu()`. Native functions use `registry.ops.my_silu()` and pass that same registry to `network_from_callable`. The shape string `x[B, ...] -> out[B, ...]` says the output has exactly the input's shape, so constraints propagate in both directions through the layer. Each node gets its own instance of the class, constructed with the resolved arguments.

Operations with several inputs, different output dimensions, or scalar arguments declare them in the same place. The built-in adaptive normalization is declared as:

```python
from hndl import Arg, operator

@operator(
    "adaptive_norm",
    summary="Instance-normalize features, then apply a per-example learned scale and bias.",
    shape="x[B, C, H, W], params[B, 2*C] -> out[B, C, H, W]",
    args={"eps": Arg(float, 1e-5, min=0, exclusive_min=True, help="Added to the variance for stability.")},
    examples=[...],
)
class AdaptiveNorm(nn.Module):
    def __init__(self, eps): ...
    def forward(self, x, params): ...
```

The shared `C` means both shapes use the same channel count; `2*C` means two style values per channel. This works in either direction: 32 feature channels require 64 style values, and 64 style values determine 32 channels. HNDL can therefore fill in an omitted style projection width before building the model. Arguments carry help text, and examples are runnable configs; `python -m hndl.docs` renders both into [docs/operators](docs/operators/index.md). See [docs/ADDING_OPERATORS.md](docs/ADDING_OPERATORS.md) for the complete format, including the `relation=` hook for rules the shape string cannot express.

Shape declarations are claims made by trusted application code. Configs only call registered names. Custom implementations still need numerical and gradient checks; a shape declaration does not prove their code correct.

## Experiment with less boilerplate

Change `linear(64)` to `linear(128)`, resolve the same input/output constraints, and send the resulting model through your existing training and evaluation loop. Compare accuracy, loss, or inference time while keeping architecture definitions small and readable. HNDL handles the connecting dimensions; your application owns metrics, optimizers, and the search over candidates.

For inspection without building a model, use the pure resolver:

```python
from hndl import resolve

plan = resolve(
    generator_config,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
)
print(plan)
```

Resolution allocates no tensors and constructs no modules. `resolve_file(...)` reads a config file; `resolve_callable(...)` captures a trusted Python function before using the same pure resolver. Save the resolved plan with your experiment to record exactly which architecture was constructed. Sequences and named branches use the same resolve-then-build workflow.

[SPEC.md](SPEC.md) defines the v1 language, registration, shape rules, and PyTorch interface. [IMPLEMENTATION.md](IMPLEMENTATION.md) describes the current alpha. HNDL is [MIT licensed](LICENSE).
