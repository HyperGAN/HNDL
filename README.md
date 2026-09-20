# HNDL

**Human-readable Network Definition Language**, pronounced “handle.”

Write your network in Python syntax, as a declarative config or a Python function. Give HNDL its input and output shapes, and it works out the connecting dimensions. Inspect what it built, access individual layers, and use the model in your PyTorch training code.

**Status: design stage.** The API and console output below describe the proposed behavior. There is no implementation yet.

## A network in a string

Use familiar calls and assignments. `x` is the input tensor and `out` selects the output. Leave out a dimension to let HNDL infer it. Input and output constraints belong in the API call:

```python
from hndl.torch import network

model = network(
    """
    hidden = linear(x, 64, name="hidden")
    hidden = relu(hidden, name="activation")
    out = linear(hidden, name="scores")
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
```

Here, `B` is a variable batch size. The final `linear` has no width argument, so the output constraint determines its 10 output features. Layer names are optional: `linear(x, 64)` works too. Variables hold tensor references; `name="hidden"` gives the layer a stable name for lookup and saved state. Reassigning `hidden` connects the next operation without changing earlier tensors. HNDL resolves the dimensions before constructing any modules.

This string is a declarative subset of Python: assignments, registered operation calls, literal arguments, and comments. HNDL parses it into a graph without executing it as Python.

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
generator_config = """
x = linear(x, name="project")
x = relu(x, name="project_relu")
x = reshape(x, 512, name="seed")
x = deconv(x, 256, policy="up2", name="up1")
x = relu(x, name="act1")
x = deconv(x, 128, policy="up2", name="up2")
x = relu(x, name="act2")
x = deconv(x, 64, policy="up2", name="up3")
x = relu(x, name="act3")
x = conv(x, 3, kernel_size=3, stride=1, padding=1, name="rgb")
out = tanh(x, name="range")
"""

generator = network(
    generator_config,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
```

The `up2` policy selects transposed-convolution settings that exactly double height and width. `reshape(x, 512)` fixes the channels; the following convolution requires an image-shaped tensor, so HNDL infers the remaining height and width. Shapes include batch, channels, height, and width:

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

Literal values remain constraints. Writing `linear(x, 128, name="project")` would also fail for the 32 × 32 target because the seed needs 8192 values. Omitted dimensions are inferred when the constraints determine them; a named policy supplies declared construction choices. Multiple valid choices are reported as ambiguous when no selected policy chooses among them. HNDL does not silently crop, broadcast, or replace layers to make them fit.

## Split and reuse tensors

Tuple assignment names the two results of a split. Pass either result into later operations, and use `#` for comments:

```python
branches = network(
    """
    # Take 64 features, then keep the remainder. dim defaults to 1.
    z1, z2 = split(x, 64, name="partition")  # Both are [B, 64] here.

    content = linear(z1, 128, name="content")
    features = relu(content, name="features")
    style = linear(z2, 128, name="style")
    out = add(features, style, name="combined")
    """,
    input_shape=("B", 128),
    output_shape=("B", 128),
    device="cpu",
)
```

`split(x, 64)` produces exactly two outputs: the first 64 entries along `dim`, and everything after them. With 160 input features, the outputs would have 64 and 96 features. Use `dim=2`, for example, to split an image's height. Both pieces must be nonempty, and splitting the batch dimension is excluded.

The proposed graph display includes every input and output port:

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

This branched model still accepts and returns a tensor. Layer lookup uses names; positional indexing and slicing apply to simple chains. The `out` binding always selects the returned tensor. A split-only config can finish with `out = z2` to return the remainder.

### Feed a branch into adaptive normalization

With the custom `adaptive_norm` operation described in the spec registered, a config can route features and style parameters:

```python
z1, z2 = split(x, 64)
features = reshape(linear(z1), 32, 4, 4)  # Projection resolves to 512.
out = adaptive_norm(features, z2)        # 32 channels need 64 style parameters.
```

Use input shape `("B", 128)` and output shape `("B", 32, 4, 4)`. This operation expects one scale adjustment and one bias per channel, so `z2` already has the required width. To learn a style mapping first, replace the final line with:

```python
style = linear(relu(linear(z2, 128)))     # Final width resolves to 64.
out = adaptive_norm(features, style)
```

Nested calls create ordinary graph nodes. A reusable subnetwork can also be registered as an operation with declared ports and shape rules, then called as `style = style_mlp(z2)`. Each call creates its own module instance when the model is built.

## Author with native Python

For reusable helpers, loops, or programmatically chosen architectures, use an ordinary Python function and the explicit trusted-code entry point:

```python
from hndl import ops
from hndl.torch import network_from_callable


def mlp(x, widths):
    for width in widths:
        x = ops.relu(ops.linear(x, width))
    return x


def architecture(x):
    z1, z2 = ops.split(x, 64, name="partition")
    content = mlp(z1, [128, 128])
    style = mlp(z2, [128])
    return ops.add(content, style, name="combined")


model = network_from_callable(
    architecture,
    input_shape=("B", 128),
    output_shape=("B", 128),
    device="cpu",
)
```

HNDL calls `architecture` once with a symbolic input to capture a finite graph. Python helpers and loops run during that construction step. The resulting model runs the captured operations on real tensors during `model(x)`; it does not rerun your authoring function. Tensor-dependent Python branching is unsupported.

Both frontends use the same operations, shape rules, resolver, and PyTorch backend. Native local variable names do not name layers; use `name="..."` when a stable layer identity matters. Unnamed calls receive generated names. The callable runs as ordinary trusted Python with your process's permissions.

## Load a config file

Save declarative assignments in a file such as `generator.hndl`, using the same syntax as `generator_config` above. Load it with constraints from your application:

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

The loader translates an explicitly allowed subset of Python's AST into graph data. It never executes config code with `eval` or `exec`, and invalid input never falls back to native Python. The proposed loader applies source, parser, graph, and model-size limits and parses all declarative input in an isolated worker. Even AST parsing can exhaust resources, so syntax restrictions alone are insufficient. See the [loading and trust contract](SPEC.md#loading-limits-and-trust-boundaries). Registered implementations remain trusted application code.

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
    max_state_bytes=0,
)
register_torch(registry, "silu", module=nn.SiLU, state_version=1)

model = network(
    """
    hidden = linear(x, 64, name="hidden")
    hidden = silu(hidden, name="activation")
    out = linear(hidden, name="scores")
    """,
    input_shape=("B", 128),
    output_shape=("B", 10),
    registry=registry,
    device="cpu",
)
```

Configs now understand `silu(x)`. Native functions use `registry.ops.silu(x)` and pass that same registry to `network_from_callable`. `preserves_shape` tells the resolver that input and output dimensions, layout, and dtype are equal, so constraints propagate in both directions. `max_state_bytes=0` declares that this operation has no parameter or buffer storage. The backend constructs an `nn.SiLU` for execution. Registration carries the operation's version; network text uses its plain name.

This helper covers unary operations with no author arguments. Operations that change shapes or accept several inputs need their own rules and port declarations, described in [SPEC.md](SPEC.md#8-custom-operators-and-minimal-graphs). Custom implementations still need numerical and gradient checks; declaring a shape rule does not prove their code correct.

## Experiment with less boilerplate

Change `linear(x, 64)` to `linear(x, 128)`, resolve the same input/output constraints, and send the resulting model through your existing training and evaluation loop. Compare accuracy, loss, or inference time while keeping architecture definitions small and readable. HNDL handles the connecting dimensions; your application owns metrics, optimizers, and the search over candidates.

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

Built-in config resolution needs no PyTorch import or tensor allocation. `resolve_file(...)` reads a config file; `resolve_callable(...)` captures a trusted Python function before using the same pure resolver. Save the resolved plan with your experiment to record exactly which architecture was constructed. Sequences, named branches, and custom multi-input operations use the same resolve-then-build workflow.

[SPEC.md](SPEC.md) defines the language, registration, shape rules, and PyTorch interface. [DESIGN.md](DESIGN.md) preserves the original rationale; the spec defines the current Python authoring and declarative loading APIs.
