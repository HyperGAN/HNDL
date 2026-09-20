# HNDL

**Human-readable Network Definition Language**, pronounced “handle.”

Describe the network you want, declare its input and output shapes, and let HNDL work out the dimensions between them. The aim is to make architectures easy to read and change, catch shape conflicts before building a model, and reduce the boilerplate around network experiments.

**Status: design stage.** This repository contains the proposed language and construction contract. The Python API below is a proposal; there is no implementation or installable package in this repository yet.

## Start with what you know

Suppose each example has 128 input features and your network should produce 10 scores. You choose a hidden layer with 64 features:

```python
from hndl import AUTO, Layer, Sequence, TensorSpec, resolve

spec = Sequence(
    schema_version=1,
    input=TensorSpec(("B", 128), layout="BF", dtype="float32"),
    output=TensorSpec(("B", 10), layout="BF", dtype="float32"),
    layers=[
        Layer("hidden", "linear@1", out_features=64),
        Layer("activation", "relu@1"),
        Layer("scores", "linear@1", out_features=AUTO),
    ],
)

plan = resolve(spec)
print(plan.describe())
```

`B` is the batch size, which can vary between calls. `BF` means batch × features. The output contract determines that `scores` needs 10 output features, and the preceding layer determines its input width. The resulting shapes are:

```text
input       [B, 128]
hidden      [B, 64]
activation  [B, 64]
scores      [B, 10]
```

Each layer has a stable name and a versioned operator, such as `linear@1`. The plan records the resolved arguments, shapes, and reasons for each inferred value. Resolution does not allocate tensors or require PyTorch.

The proposed PyTorch backend then builds a normal `torch.nn.Module`:

```python
import torch
from hndl.torch import build

model = build(plan, device="cpu", initialization_seed=7)
x = torch.randn(8, 128)
scores = model(x=x)["output"]  # Shape: [8, 10]
```

The host application supplies tensors and chooses the device. Every call returns a dictionary of named outputs, including for a single-output network. Training uses ordinary PyTorch parameters, autograd, and optimizers.

## Change the target, resolve the dimensions

A generator makes the idea more useful. Start with a 128-value vector and ask for a 32 × 32 RGB image. Choose three upsampling stages and their channel counts; leave the projection width and starting spatial dimensions automatic.

```python
from hndl import AUTO, Layer, Policy, Sequence, TensorSpec, resolve

spec = Sequence(
    schema_version=1,
    input=TensorSpec(("B", 128), layout="BF", dtype="float32"),
    output=TensorSpec(("B", 3, 32, 32), layout="NCHW", dtype="float32"),
    policies=[
        Policy("spatial.up2_transpose@1", nodes=["up1", "up2", "up3"]),
    ],
    layers=[
        Layer("project", "linear@1", out_features=AUTO),
        Layer("project_relu", "relu@1"),
        Layer("seed", "reshape@1", shape=[512, AUTO, AUTO]),
        Layer("up1", "conv_transpose2d@1", out_channels=256),
        Layer("act1", "relu@1"),
        Layer("up2", "conv_transpose2d@1", out_channels=128),
        Layer("act2", "relu@1"),
        Layer("up3", "conv_transpose2d@1", out_channels=64),
        Layer("act3", "relu@1"),
        Layer("rgb", "conv2d@1", out_channels=3,
              kernel_size=3, stride=1, padding=1),
        Layer("range", "tanh@1"),
    ],
)

plan = resolve(spec)
```

`NCHW` means batch × channels × height × width. The selected policy fixes each transposed convolution to kernel 4, stride 2, padding 1, dilation 1, output padding 0, and groups 1. These settings exactly double height and width. Three stages therefore require a 4 × 4 starting map to reach 32 × 32, so the projection must produce `512 × 4 × 4 = 8192` values.

Edit the output contract and resolve a new plan:

| Target image | Starting map | Projection width |
| --- | --- | --- |
| 32 × 32 | 512 × 4 × 4 | 8192 |
| 64 × 64 | 512 × 8 × 8 | 32768 |
| 32 × 64 | 512 × 4 × 8 | 16384 |

The layer sequence stays the same. Each new plan describes a new model with its own parameter shapes. `tanh` is an explicit architecture choice; your data preparation and objective must use the intended output range.

A 30 × 30 target cannot satisfy those three exact doubling stages. Resolution should report an error along these lines before any model is built:

```text
E_CONSTRAINT: output height 30 conflicts with up1 → up2 → up3.
Three doubling stages require an integer seed height; 30 / 8 = 3.75.
Choose a target divisible by 8 or explicitly change the architecture.
```

Likewise, setting `project.out_features=128` for the 32 × 32 target produces a reshape conflict: the starting map needs 8192 values. HNDL preserves the value you wrote and explains the conflict.

## What automatic means

- **Literal values are constraints.** A width of `64` remains `64`.
- **`AUTO` asks for resolution.** HNDL can infer a value when the constraints determine it uniquely.
- **Named policies make declared choices.** The doubling policy above supplies exact convolution settings to the selected layers.
- **Unanswered choices are errors.** If several shapes are valid and no selected policy decides between them, the plan remains ambiguous.

HNDL will not silently crop an image, insert a projection, broadcast a join, or change a layer to make the shapes fit. Omitted optional arguments use documented defaults unless an explicitly selected policy supplies them; `AUTO` explicitly requests resolution and is not a synonym for an omitted argument.

Shape checks cover the declared contracts and supported operator rules. They cannot establish model quality or the correctness of arbitrary custom code. Runtime input checks still validate the tensors your application supplies.

## Keep experiments focused on the network

The intended workflow is to edit a specification, resolve and inspect it, build the model, and run your existing training or evaluation code. Compare candidate architectures using the metrics you care about, then save the specification and resolved plan alongside the experiment results.

For example, try hidden widths of 32, 64, and 128 in the first example. Resolve and build each candidate, then pass it to the same training and evaluation function in your application. Compare validation accuracy and inference time. The input stays at 128 features and the output stays at 10 scores; HNDL resolves the connecting dimensions for each candidate.

HNDL handles architecture constraints and module construction. Your application owns datasets, losses, metrics, optimizers, training schedules, and any search over candidate networks. A valid shape plan is a useful starting point for an experiment, not a prediction that its metrics will improve.

The proposed v1 also includes structured graphs with named inputs and outputs, branches and joins, and custom operators with declared shape rules. These support residual paths and style-conditioned blocks while keeping the same resolve-then-build workflow. A compact sequence syntax is planned after the structured core.

## Read the contract

- [SPEC.md](SPEC.md) defines the technical v1 contract: data structures, resolution rules, operators, policies, plans, runtime behavior, and acceptance criteria.
- [DESIGN.md](DESIGN.md) records the original design and its rationale.

Implementation begins with the pure resolver and shape checks, followed by the PyTorch builder and graph extensions. HNDL is intended to work independently of any training framework, including HyperGAN.
