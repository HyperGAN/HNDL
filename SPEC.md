# HNDL v1 technical specification

**Status: proposed; no implementation exists yet.** This document defines the baseline for implementing **HNDL — Human-readable Network Definition Language**, pronounced “handle.” It turns the direction in [DESIGN.md](DESIGN.md) into a standalone technical contract. [README.md](README.md) introduces the same ideas with examples.

“Must” denotes a v1 requirement. Python signatures and JSON examples describe the proposed interface, not an available package. This specification supersedes conflicting public API descriptions in DESIGN.md: the primary authoring surface is a multiline network DSL plus input/output constraints passed through Python. Open interface decisions are listed at the end; implementation must settle those before dependent features or serialized formats ship.

## 1. Purpose and boundary

HNDL describes a neural network separately from its training code. Given input/output tensor contracts, a partially specified architecture, and explicitly selected construction policies, it resolves concrete dimensions or explains why resolution failed. It then builds ordinary PyTorch modules from the resolved plan.

```text
Author specification + tensor contracts + registered rules/policies
                              |
                           resolve
                              |
                Concrete plan + resolution trace
                              |
                            build
                              |
                     Ordinary torch module
```

The initial deliverable includes the sequence DSL, a typed Python construction API, bounded shape inference, a PyTorch backend, layer-by-layer inspection, and custom operator registration. A JSON-compatible graph representation underlies resolution and supports the minimal custom multi-input extension. Object-based layer helpers are implementation details, not the primary public interface.

HNDL owns tensor contracts, architecture resolution, construction, module registration, plan persistence, and diagnostics. The host owns data semantics, losses, metrics, regularizers, optimizers, training schedules, runtime random streams, devices, and complete checkpoint recovery. Researchers can vary architecture specifications while keeping their metric evaluation loop unchanged; HNDL does not optimize a metric or search architectures automatically.

Topology search, dynamic image sizes, runtime control flow, weight tying between graph nodes, distributed training, export backends, a visual editor, and full StyleGAN recipes are outside v1. HNDL has no HyperGAN runtime dependency.

## 2. Core invariants

1. **Literal values are constraints.** Resolution must never replace a specified width, stride, kernel, activation, or binding to make a network fit.
2. **Inference is not selection.** Uniquely determined values may be inferred. Choosing among valid architectures requires a named policy or more explicit information.
3. **Resolution is pure.** Built-in validation, resolution, and serialization require no torch import, tensor allocation, CUDA, random sampling, or network access.
4. **Successful plans are concrete.** All construction dimensions and operator arguments are resolved. Only the declared batch symbol may remain variable.
5. **Every resolved value has provenance.** Record whether it was explicit, an operator default, inferred, or policy-selected, including the contributing rule or source.
6. **Forward execution uses the plan.** It does not parse strings, solve constraints, select policies, or create parameters.
7. **Limitations are visible.** Ambiguity, unsupported inference, contradictions, and asserted custom contracts must remain distinguishable.

## 3. Tensor contracts and automatic values

A `TensorSpec` contains ordered dimensions including batch, an explicit layout, and a dtype. Initially supported layouts are:

| Layout | Dimensions | Meaning |
| --- | --- | --- |
| `BF` | `[B, F]` | Batch and feature width |
| `NCHW` | `[B, C, H, W]` | Batch, channels, height, width |

Qualified v1 execution uses `float32`. A shared batch symbol, written `"B"` in these examples, enforces equality across participating inputs and outputs. Concrete dimensions and runtime batch sizes must be positive integers. Only batch may remain symbolic after resolution; symbolic batch is preserved through reshape and flatten.

The sequence API accepts `input_shape` and `output_shape`, both including batch. Rank two means `BF`; rank four means `NCHW`. Unsupported ranks fail rather than guessing a layout. The `dtype` keyword defaults to `"float32"`. Structured graph contracts retain explicit layouts.

The DSL token `auto` marks an explicitly unresolved numeric field, represented internally as `AUTO` and serialized as `{"auto": true}`. It is valid only where the operator schema permits it. It is not interchangeable with `None`, `null`, `-1`, zero, or an arbitrary expression.

| Author value | Meaning |
| --- | --- |
| Literal | A hard constraint; a mismatch fails |
| `AUTO` | Resolve through supported relations or an eligible selected policy; otherwise fail |
| Omitted optional field | Use its versioned operator default, unless reserved for an applicable policy |
| Omitted required field | Fail, unless a selected policy explicitly supplies that field |

`AUTO` must not quietly become an operator default when inference stalls. Derived input widths/channels are operator relations, not additional required user arguments.

HNDL does not infer color space, value range, activation choice, layout conversion, mixed precision, or broadcasting. A declared `tanh` is an explicit operation; an image output contract does not imply it. Joins must not insert resizing or broadcasting.

## 4. Author specification and graph

The canonical internal model is a finite directed acyclic graph. The sequence DSL lowers into this graph before resolution. Structured graph input remains available for named multi-input/multi-output networks.

| Field | Contract |
| --- | --- |
| `schema_version` | Author data-format version; initially `1` |
| `inputs` | Named external `TensorSpec` contracts |
| `nodes` | Ordered node declarations with stable `id`, versioned `op`, `args`, and input-port `inputs` bindings |
| `outputs` | Named output references with required tensor contracts |
| `policies` | Exact policy IDs/versions, arguments, and explicit node selections |
| `metadata` | Optional annotations excluded from numerical identity |

Input, output, and node IDs match `[a-z][a-z0-9_]*`. References are `input:<name>` or `node:<id>/<port>`. Built-in unary nodes consume `x` and produce `out`. Custom operators declare every input and output port.

The following proposed internal/structured JSON encoding defines a two-layer perceptron whose last width is resolved from the output contract. Ordinary sequence users write the DSL in §7 instead. Container/key choices shown here are the baseline for the implementation; a machine-readable schema is still required before release.

```json
{
  "schema_version": 1,
  "inputs": {
    "x": {"shape": ["B", 128], "layout": "BF", "dtype": "float32"}
  },
  "nodes": [
    {
      "id": "hidden",
      "op": "linear@1",
      "args": {"out_features": 64, "bias": true},
      "inputs": {"x": "input:x"}
    },
    {
      "id": "act",
      "op": "relu@1",
      "args": {},
      "inputs": {"x": "node:hidden/out"}
    },
    {
      "id": "head",
      "op": "linear@1",
      "args": {"out_features": {"auto": true}, "bias": true},
      "inputs": {"x": "node:act/out"}
    }
  ],
  "outputs": {
    "output": {
      "ref": "node:head/out",
      "spec": {"shape": ["B", 10], "layout": "BF", "dtype": "float32"}
    }
  },
  "policies": [],
  "metadata": {}
}
```

Validation must reject duplicate IDs, missing references, undeclared ports, missing required bindings, cycles, and nodes that cannot reach any declared output. Multiple public outputs are allowed. Graph values are tensors; adapters must flatten Python tuple/dict results into declared tensor ports.

Each node owns an independent module instance and executes once per forward. Fan-out reuses a computed tensor. It does not clone the source module, repeat its invocation, or tie parameters between nodes. Explicit weight tying between graph nodes is deferred and must fail if requested. External reuse of built modules through sequence slices is supported as described in §7.

The sequence DSL lowers to external input `x`, public output `output`, and consecutive unary nodes connected through `out`. Structured authoring requires explicit node IDs. The DSL may generate deterministic position-based IDs for unlabeled lines but must show them in the expansion and warn that inserting layers changes later generated names. Explicit labels become stable node IDs.

## 5. Resolution algorithm

The resolver must perform the following logical steps:

1. Validate the schema and graph; bind exact operator and policy versions from the active registry.
2. Expand explicitly declared finite policy templates, if any, and validate their nodes/edges. The required initial policy does not change topology.
3. Seed tensor contracts and literals. Apply policy requirements as constraints. Apply operator defaults only to omitted fields not already supplied by policy requirements or reserved for selected policy defaults. Create operator relations and automatic variables.
4. Propagate forward and backward to a fixed point, recording derivations and detecting contradictions.
5. Ask selected policies to resolve remaining eligible choices according to their deterministic rules, then propagate again.
6. Fail if any required construction value remains ambiguous or unresolved; otherwise freeze the concrete plan.

Rules may only add or refine facts. They must not retract constraints or overwrite literals. The resolver must bound expansion, dimension products, candidate sets, and custom-rule execution. Exceeding a configured limit is an explicit failure, never silent truncation. Exact limits/configuration APIs remain an implementation decision.

The required relation vocabulary includes positive integers, equality, multiplication by a constant, products, divisibility, and fixed convolution equations. Direct equations, finite candidates, and bounded interval reasoning are sufficient; a general symbolic or SMT solver is not required.

Three outcomes must stay distinct:

| Outcome | Meaning |
| --- | --- |
| Contradiction | The supported relations prove the declared constraints incompatible |
| Ambiguous | Multiple valid values remain and no selected policy chooses among them |
| Unresolved | Available rules cannot determine a required value |

An unsupported inverse is not proof that no architecture exists. A non-unique integer interval must not be resolved by picking its smallest member unless a selected policy explicitly requires that choice.

### Policies

A policy has an immutable ID/version, applicability, eligible fields, deterministic selection rules, and optional validation requirements. It must distinguish hard **requirements** from **defaults**. Requirements enter before inference; they can conflict with a literal and must then fail.

For eligible default choices, precedence is explicit value → uniquely inferred value → policy-selected value → applicable operator default. Fields reserved for policy choice must not be prematurely fixed by an operator default. Conflicting policy assignments fail; policy order must not determine the result. All effects are recorded in the plan.

The required initial policy is `spatial.up2_transpose@1`, selected in the DSL as `policy=up2` through the built-in registry alias:

| Property | Requirement |
| --- | --- |
| Applies to | Explicitly selected `conv_transpose2d@1` nodes |
| `kernel_size` | `4` on both spatial axes |
| `stride` | `2` on both axes |
| `padding` | `1` on both axes |
| `dilation` | `1` on both axes |
| `output_padding` | `0` on both axes |
| `groups` | `1` |

It supplies omitted/automatic eligible values and rejects contrary literals. These settings guarantee exact spatial doubling; channel widths and bias are separate arguments. The policy does not guarantee image quality. It must not silently substitute resize/convolution, crop, or a different architecture when a target fails.

Custom policies are trusted, explicitly registered Python providers. Data files cannot contain executable expressions or arbitrary imports.

## 6. Built-in operators

The minimum catalog is listed below; each identifier has semantic version `@1`. Unary operators use ports `x`/`out` unless stated otherwise. All propagate batch equality and preserve dtype. Equations describe shape relations, not a license to accept arguments that the backend rejects.

| Operator | Arguments/ports relevant to shape | Required relation |
| --- | --- | --- |
| `linear` | `out_features`, optional bias | `BF → BF`; infer input width, constrain output width; no implicit flatten |
| `reshape` | Non-batch `shape` | Preserve per-example element count and batch; output must have a supported layout |
| `flatten` | No shape choices | Combine all non-batch dimensions into `BF` |
| `conv2d` | `out_channels`, kernel, stride, padding, dilation, groups | `NCHW → NCHW`; use the formula below independently per spatial axis |
| `conv_transpose2d` | As above, plus `output_padding` | `NCHW → NCHW`; use the transpose formula below |
| `relu` | None affecting shape | Preserve dimensions/layout; out-of-place |
| `leaky_relu` | Negative slope | Preserve dimensions/layout; out-of-place |
| `tanh` | None affecting shape | Preserve dimensions/layout; out-of-place |
| `group_norm` | Explicit positive group count | Preserve shape; groups must divide channels |
| `add` | Ports `a`, `b`; output `out` | Shapes, layout, dtype, and batch must match exactly |
| `concat` | Declared input count `K`, axis, ports `x0` through `x(K-1)` | Sum the selected non-batch axis; all other axes/layout/dtype must agree |

For each spatial axis, with input extent `I`, output extent `O`, kernel `k`, stride `s`, padding `p`, and dilation `d`:

```text
conv2d:           O = floor((I + 2p - d(k - 1) - 1) / s + 1)
conv_transpose2d: O = (I - 1)s - 2p + d(k - 1) + output_padding + 1
```

Dimensions, channels, kernel sizes, strides, and dilations must be positive. Convolution padding/output padding must be nonnegative; group divisibility and backend argument restrictions must also hold. The complete versioned argument schema must publish the allowed scalar/pair forms, defaults, and automatic fields before these operators ship.

Defaults required by the examples are fixed for semantic version 1: `linear.bias=true`; convolution `bias=true`, `stride=1`, `padding=0`, `dilation=1`, and `groups=1`; transposed convolution additionally has `output_padding=0`. Convolution `kernel_size` and output channels are required unless a selected policy supplies an eligible field. Linear output width is required but may be `auto`. The `up2` policy's requirements take precedence over these defaults. Built-in activations are out-of-place. These are HNDL versioned choices, not dynamically inherited backend defaults.

Ordinary convolution inversion may produce an integer interval, not a unique input. Transposed convolution does not inherently mean doubling. `reshape` may directly solve exactly one unknown product factor; multiple unknown factors require independent relations or a selected policy. Each `AUTO` occurrence is a distinct unknown unless a supported relation connects it. For example, known target height and width may separately determine two automatic seed axes.

For concat, axes index the complete tensor shape and axis `0` is batch, which cannot be concatenated in v1. The spelling/normalization of negative axes remains to be fixed in the argument schema. No join introduces broadcasting, casts, or layout conversion.

## 7. DSL and public API

The primary API accepts a multiline string with one layer per nonblank line and tensor constraints supplied separately:

```python
from hndl.torch import network

source = """
hidden: linear 64
activation: relu
scores: linear auto
"""
model = network(
    source,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
    initialization_seed=7,
)
print(model.describe())
# With a caller-supplied tensor x of shape [B, 128]:
# scores = model(x)
```

`describe()` returns a printable table with network input/output shapes and dtype, followed by every layer's index, name, operation, and resolved input/output shapes. The following console output is illustrative; no implementation has produced it yet:

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name        operation  input shape  output shape
0      hidden      linear     [B, 128]     [B, 64]
1      activation  relu       [B, 64]      [B, 64]
2      scores      linear     [B, 64]      [B, 10]
```

The public baseline is:

| API | Result and requirements |
| --- | --- |
| `resolve(source, *, input_shape, output_shape, dtype="float32", registry=None)` | Parse and resolve a sequence into an immutable `ResolvedPlan`, without torch |
| `network(source, *, input_shape, output_shape, device, initialization_seed, dtype="float32", registry=None)` | Resolve and build a sequence module with tensor-returning `forward(x)` |
| `model.plan` | The immutable resolved plan used to construct the module |
| `model.describe()` | A complete printable layer input/output shape table; must work without sample tensors or a forward pass |
| `plan.describe()` | Detailed node/port shapes, value provenance, asserted contracts, static parameter counts where known, and semantic digest |
| `hndl.torch.build(plan, *, device, initialization_seed)` | Lower-level construction returning a dictionary-output `GraphModule` |

`resolve` is exported by `hndl`; `network` is exported by `hndl.torch`. The latter is the resolve/build convenience API and must not introduce different resolution semantics. A missing registry selects the built-in registry. Structured graph resolution and serialization helpers remain to be named; their data contracts are defined here.

### Sequence syntax and exact version bindings

A line contains an optional `id:`, an operator alias, documented positional arguments, and named scalar arguments. Blank lines and surrounding whitespace are ignored. Labels follow the node-ID rules in §4. `auto`, `true`, and `false` are lowercase. Allow integers, finite numeric values, and registered aliases for policy selection. Unknown operators, policies, arguments, duplicate labels, and repeated positional/named assignments fail with source locations.

The initial positional mappings are `linear <out_features>`, `conv <out_channels>`, `deconv <out_channels>`, and `reshape <non-batch dimensions...>`. `relu`, `tanh`, and `flatten` require no positional arguments. Other scalar arguments use their operator schema's field names, such as `kernel_size=3`. Complex/tuple-valued custom arguments use structured data until separately specified. Multi-input joins use the graph representation, not implicit names inside sequences.

The DSL does not use `@` version suffixes. Registry bindings map aliases to exact immutable identities: `linear` to `linear@1`, `conv` to `conv2d@1`, `deconv` to `conv_transpose2d@1`, and policy `up2` to `spatial.up2_transpose@1`. Other built-in aliases bind the matching catalog operator at version 1. Custom aliases use explicit registration (§8). Duplicate/conflicting aliases fail. Resolution must never choose a newest installed version or discover arbitrary plugins. Saved plans materialize exact identities/versions and the effects of policy selection, so restoration does not depend on current alias bindings.

There is no Python evaluation, arithmetic, loops, nesting, implicit tensor context, or imports in the DSL. Its semantics must match equivalent structured graph input. Exact lexical details beyond this baseline must be fixed in the parser schema before shipping.

### Layer access

The sequence module provides access to the actual registered PyTorch layers:

```python
first = model[0]
last = model[-1]
hidden = model["hidden"]
assert first is hidden
assert len(model) == 3
layers = list(model)
features = model[:2]
```

Integer indexing, including negative indices, follows layer order; string lookup uses the node ID. `len(model)` counts layers and iteration yields those same modules in order. An unknown name raises `KeyError`; an out-of-range integer raises `IndexError`.

A slice returns an ordinary `torch.nn.Sequential` containing the selected existing module objects in slice order. It shares parameters, buffers, training mode, and device changes with the original modules; it is not a copy. The sliced object has no HNDL plan or shape guarantees, and arbitrary slices may not form a shape-compatible sequence. Creating a slice must not add registrations to the original model or alter its state keys.

The facade must preserve the backend's single registration under `nodes.n_<node_id>` and corresponding state keys, without registering each layer a second time or adding a wrapper prefix. Parameter access, gradients, optimizers, state dictionaries, device moves, and train/eval use ordinary PyTorch behavior. Structural replacement, insertion, or deletion through the HNDL container is rejected to keep its plan sound; ordinary parameter value updates remain supported. Architecture changes require resolving/building a new model.

General graphs support node-name lookup only; positional indexing, slicing, and sequence iteration are not graph APIs. Their execution interface remains `forward(**inputs)` returning a dictionary of named tensors.

### Worked generator resolution

The generator uses three explicit doubling stages to work backward from an output contract:

```python
from hndl.torch import network

source = """
project: linear auto
project_relu: relu
seed: reshape 512 auto auto
up1: deconv 256 policy=up2
act1: relu
up2: deconv 128 policy=up2
act2: relu
up3: deconv 64 policy=up2
act3: relu
rgb: conv 3 kernel_size=3 stride=1 padding=1
range: tanh
"""
model = network(
    source,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
    initialization_seed=7,
)
print(model.describe())
# With a caller-supplied tensor z of shape [B, 128]:
# image = model(z)
```

Illustrative complete shape output:

```text
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

The last convolution preserves spatial size. The selected transpose policy gives `output_height = 8 * seed_height` and the same relation for width. The projection width is then `512 * seed_height * seed_width`.

| Required output | Resolved seed | `project.out_features` | Outcome |
| --- | --- | --- | --- |
| `[B,3,32,32]` | `[B,512,4,4]` | `8192` | Resolved |
| `[B,3,64,64]` | `[B,512,8,8]` | `32768` | Resolved |
| `[B,3,32,64]` | `[B,512,4,8]` | `16384` | Resolved |
| `[B,3,30,30]` | Would require axes of `3.75` | — | Contradiction |
| `[B,3,32,32]` with literal projection width `128` | Needs `8192` elements | — | Contradiction; retain the literal |

## 8. Custom operators and minimal graphs

A custom operator registration must provide:

| Component | Requirement |
| --- | --- |
| Identity | Namespaced ID and immutable semantic version; duplicate registration fails |
| Arguments | Types, required/default values, automatic eligibility, validation limits |
| Ports | Named tensor inputs/outputs and ordering where applicable |
| Constraints | Pure supported relations or bounded refinements with reasons |
| Builder | Normal `nn.Module` built from concrete arguments without hidden globals/network access |
| Invocation adapter | Bind input ports to arguments and map results back to output ports |
| State | Stable parameter/buffer names and a state compatibility version |
| Capabilities | Declared/tested dtype/device and gradient behavior, separate from host qualification |

Shape providers and backend builders may live in separate modules. Resolving a data specification must not import arbitrary plugins or execute its contents. Registered providers are trusted code; opaque modules may instead use fully declared **asserted** input/output contracts. Those contracts are not statically verified and require runtime checks before training. Backward inference through an opaque module is unavailable unless its provider supplies the necessary relations.

### Registering a unary shape-preserving layer

The simplest extension registers its pure shape rule separately from its PyTorch implementation:

```python
from torch import nn
from hndl import Registry, preserves_shape
from hndl.torch import network, register_torch

registry = Registry.builtins()
registry.register(
    "silu", identity="example.silu", version=1, shape=preserves_shape,
)
register_torch(registry, "silu", module=nn.SiLU, state_version=1)

model = network(
    """
hidden: linear 64
activation: silu
scores: linear auto
""",
    input_shape=("B", 128),
    output_shape=("B", 10),
    registry=registry,
    device="cpu",
    initialization_seed=7,
)
```

`Registry.builtins()` returns an independently extensible registry containing the fixed built-in aliases. `register(alias, *, identity, version, shape, state_version=1)` binds a new alias to its exact operator identity/version. The no-argument unary helper shown here supplies an empty argument schema, input port `x`, output port `out`, default state compatibility version `1`, and unqualified capability status. State compatibility metadata belongs to the pure registration, so resolution can record it without loading a backend. More complex operators must declare their schemas/ports explicitly through the full extension API.

`preserves_shape` contributes bidirectional equality relations for input/output dimensions, layout, and dtype; it is not merely a forward shape callback. Downstream constraints can therefore propagate through this custom layer. The provider's claim must still be verified against the actual module during numerical qualification; registering it does not qualify its device/dtype/gradient behavior.

`register_torch(registry, alias, *, module, state_version)` attaches a backend constructor to the alias's already registered exact identity/version. Missing bindings, duplicate backend registration, and a state version differing from the pure declaration fail. In the no-argument unary form, the builder constructs `module()` once per node, invokes it with the `x` tensor, and binds its tensor result to `out`. There is no constructor introspection or assumption about other module arguments. `nn.SiLU` here uses its ordinary non-in-place constructor behavior.

Pure registration and `resolve()` remain usable without importing torch, even though the combined example imports torch to register a backend. Resolution must not call the backend constructor. Backend bindings stay separate from serializable arguments; neither DSL nor saved data can import a module class. The plan stores `example.silu@1` and its state compatibility requirement, not executable Python. Building without its explicitly registered backend must fail.

### Required affine instance-normalization fixture

v1 must demonstrate a small style-conditioned graph: a mapping MLP produces `w`; `w` fans out into feature projection/reshape and a style affine; both feed a custom adaptive normalization node.

For a concrete fixture, declare external input `z: [B,128]` and output `features: [B,64,4,4]`. The following nodes show the port bindings and resolved shapes; `adaptive_norm@1` is shorthand for the fixture's explicitly registered custom operator.

| Node | Operator and author arguments | Input bindings | Resolved output |
| --- | --- | --- | --- |
| `mapping` | `linear@1`, `out_features=256` | `x=input:z` | `[B,256]` |
| `w` | `relu@1` | `x=node:mapping/out` | `[B,256]` |
| `project` | `linear@1`, `out_features=AUTO` | `x=node:w/out` | `[B,1024]` |
| `seed` | `reshape@1`, `shape=[64,4,4]` | `x=node:project/out` | `[B,64,4,4]` |
| `style` | `linear@1`, `out_features=AUTO` | `x=node:w/out` | `[B,128]` |
| `norm` | Custom `adaptive_norm@1` | `x=node:seed/out`, `params=node:style/out` | `[B,64,4,4]` |

Bind public output `features` to `node:norm/out`. Both branches consume the same `w` tensor; `mapping` and `w` each execute once. Reshape determines the projection width as `64 * 4 * 4`; the custom relation below determines the style width as `2 * 64`.

The fixture operator, referred to as `adaptive_norm@1` here, has ports `x`, `params`, and `out`:

```text
x:      [B, C, H, W]   NCHW
params: [B, 2*C]       BF
out:    [B, C, H, W]   NCHW
```

The relation `params.features = 2 * x.channels` must resolve an automatic style-affine output width. Batch/dtype equality and output-shape equality are required. Its exact numerical convention is:

```text
mean       = mean(x, spatial_axes, keepdim=True)
variance   = mean((x - mean)^2, spatial_axes, keepdim=True)
normalized = (x - mean) / sqrt(variance + eps)
[delta_gamma, beta] = split(params, C, axis=features)
out = (1 + delta_gamma)[B,C,1,1] * normalized + beta[B,C,1,1]
```

`eps` must be positive, defaults to `1e-5` for this fixture version, and is explicit in the plan. Variance is population variance. Both feature and style paths are differentiable. The operation has no running statistics and identical train/eval behavior. The fixture explicitly zero-initializes its style affine, giving unit scale and zero bias on normalized features; arbitrary linear layers do not inherit that initializer.

This is an AdaIN-style extension demonstration, not a faithful StyleGAN implementation. Learned constants, noise blocks, and modulated convolutions are separate future operators. Future stochastic operators should receive caller-owned noise through tensor ports.

## 9. Resolved plans and reproducibility

A `ResolvedPlan` must be immutable and include:

- Author/plan schema and resolution-semantics versions; exact operator and policy versions.
- Concrete nodes, edges, ports, arguments, tensor contracts, and verified/asserted status.
- Expanded initialization/trainability settings and stable execution order.
- Resolution trace, source locations, and value provenance.
- Semantic/artifact digests, implementation/dependency provenance, and state compatibility requirements.

Serialization uses canonical JSON with sorted object keys, ordered node lists, normalized tuple/numeric representations, no non-finite values, and SHA-256. Exact encoding and digest test vectors must be published before serialized plans ship.

The **semantic digest** covers concrete graph structure, state IDs, operator versions, arguments, contracts, initialization, trainability, and execution order. It excludes comments, source locations, and inference explanations. Policy provenance is preserved, but after effects are expanded, equivalent concrete numerical plans must have identical semantic identity regardless of authoring syntax or derivation.

The **artifact digest** covers the complete saved plan except its own digest field. It includes provenance and trace information that can differ between numerically equivalent plans. Parameter counts, where statically available, are not estimates of peak training memory.

## 10. PyTorch backend contract

`build(plan, *, device, initialization_seed)` returns a normal training-mode `GraphModule`. `network(...)` presents the sequence facade from §7 over the same construction semantics and registered state. Device is explicit; CPU is suitable for small fixtures and CUDA is the intended primary execution target. There is no silent CPU fallback. The plan fixes the qualified dtype, initially `float32`.

- A `GraphModule` validates declared external input contracts in `forward(**inputs)` and returns a dictionary keyed by public output names, including for a single output. The sequence facade instead accepts `forward(x)` and returns the single output tensor, preserving the same contract checks.
- Modules register once under `nodes.n_<node_id>`; the prefix avoids collisions with module attribute names. Stateless nodes keep execution/diagnostic identities without state entries.
- Nodes run in stable topological order, breaking ties by declaration order. The order is saved in the plan.
- Parameters/buffers are fully materialized before optimizer or distributed setup. No first-forward parameter creation is allowed.
- Runtime must preserve ordinary gradients. It must not silently detach, clone, cast, move tensors, or mutate a shared branch through in-place built-in activations.
- Contract mismatches must be reported before applying the affected layer where possible. Asserted custom contracts require runtime validation.

Initialization belongs to the plan and build receipt. Initial support is versioned `torch_default@1` plus explicit constant overrides. The supported PyTorch runtime must be recorded because its defaults affect initialization. A supplied seed uses an isolated RNG scope on the requested device and restores the caller's RNG state. Module construction order is stable; custom builders must respect the same requirements. Exact initial tensors across devices or PyTorch versions are not promised.

Per-node trainability masks are applied during construction and saved in the plan. Freezing parameters does not select evaluation mode. The host controls train/eval behavior and must preserve these masks. Stateful custom modules declare persistent buffers and manage derived caches correctly after state loading/device moves.

## 11. Persistence and compatibility

Persist the author specification and full resolved plan. Reconstruct from the saved plan, not by applying a newer resolver to old `AUTO` fields. Before state loading, validate the semantic digest, required implementation/state versions, and state names/shapes/dtypes.

These identities serve different purposes:

| Identity/version | Purpose |
| --- | --- |
| Package version / implementation revision | Identify implementation provenance |
| Author and plan schema versions | Decode saved structures |
| Resolution-semantics version | Identify planning behavior |
| Operator/policy semantic versions | Preserve mathematical/construction meaning |
| Module state compatibility version | Identify compatible parameter/buffer layouts and behavior |
| Host checkpoint compatibility version | Decide complete training continuation compatibility |

An older semantic version must retain its meaning or fail with the affected version and an actionable compatibility/migration explanation. A package version alone is not proof of numerical compatibility. An architecture change requires a new model or explicit weight transfer; it is not ordinary checkpoint continuation.

Adapters for host applications may select one named output tensor and save the HNDL plan alongside training state. Optimizer, EMA, RNG, data position, distributed execution, and complete recovery remain host responsibilities. Construction success does not qualify a network for a host's numerical or distributed guarantees.

## 12. Diagnostics

Failures must expose a stable code, source/node/field location, affected constraint chain, and suggested explicit remedies. Suggestions must never modify the specification automatically.

| Code | Example |
| --- | --- |
| `E_CONSTRAINT` | Target `30` conflicts with three exact doubling stages: seed would be `3.75` |
| `E_RESHAPE` | Literal projection width `128` cannot supply required `512 * 4 * 4 = 8192` elements |
| `E_AMBIGUOUS` | A convolution inverse leaves several valid input sizes |
| `E_UNRESOLVED` | A custom block lacks the relation needed for backward inference |
| `E_POLICY_CONFLICT` | Literal stride `1` conflicts with the selected doubling policy |
| `E_BINDING` | A reference names an undeclared node or tensor port |
| `E_STATE_VERSION` | Saved state requires an unavailable compatible operator implementation |

Schema/version/resource-limit diagnostics also require stable codes; their complete catalog is pending. DSL failures include line/column locations where applicable. `model.describe()` must include every layer, its ID/operator, and complete input/output shapes without executing the network. `plan.describe()` must additionally expose provenance sufficiently to explain why a field changed between separately resolved specifications.

Static resolution, dry-run/meta checks, and numerical preflight are distinct inspection modes. Numerical preflight must use disposable modules or restore affected state/RNG; it must not advance a live training stream or alter persistent buffers.

## 13. Acceptance criteria and implementation order

The following are future gates, not claims about tests already passing:

| Gate | Required evidence |
| --- | --- |
| Pure core | Import, parse DSL, resolve, and serialize with no torch/CUDA; malformed specs and unavailable providers fail clearly |
| Shape resolution | Generator targets `32×32`, `64×64`, `32×64`; invalid `30×30`; literal-width conflict; inverse ambiguity; group divisibility and join conflicts |
| Determinism | Repeated resolution is identical; equivalent DSL/structured graph inputs have the same semantic plan; aliases bind exact versions; policy order has no effect |
| Backend equivalence | Handwritten PyTorch comparison from identical state: forward values, input/parameter gradients, optimizer updates |
| Public sequence API | Tensor forward result; complete shape table; integer/negative/name lookup and iteration; shared-module slices; stable state keys without double registration; structural assignment rejected |
| Unary registration | `silu` example resolves through bidirectional shape equality; pure resolution never constructs/imports a backend; exact identity/state version is preserved |
| Custom graph | Resolve `2*C`; fan-out works; feature/style gradients are correct; reject incorrect batch/channel contracts |
| Numerical validity | Finite affine-normalization forward/backward for constant/nonconstant inputs; first/second derivative checks on suitable nondegenerate fixtures |
| Registration and recovery | Stable registered state; fresh-process round-trip; no first-forward parameters or hidden runtime RNG draws |
| CUDA | Selected operators and custom fixture execute and reload on CUDA within declared tolerances |
| Compatibility | Load compatible saved plans without replanning; reject incompatible semantics/state versions actionably |

No identity substitutes, blanket skips, fabricated successful output, or silent device fallback can satisfy these gates. CPU correctness does not establish CUDA correctness. Host integration, complete checkpoint recovery, two-GPU behavior, and multi-host behavior are separate integration gates.

Implement in this order:

1. Pure schema/types, fixed registry aliases, the sequence DSL parser, linear/reshape/activation relations, trace, and plan serialization.
2. Convolution relations, doubling policy, complete shape inspection, and DSL/graph semantic equivalence and determinism checks.
3. PyTorch `network` facade and lower-level builder, layer access/slicing, state registration, isolated initialization, CPU equivalence, and CUDA checks.
4. Simple custom-layer registration, then named graph joins/fan-out, full custom providers, and the affine-normalization fixture.
5. Complete the public DSL walkthroughs and acceptance gates; host adapters follow the standalone core.

A sequence-only first milestone is useful progress, not completion of the graph/custom-extension v1 contract. Host adapters follow the standalone core.

## 14. Decisions to settle before implementation depends on them

The baseline deliberately leaves these details visible:

- Publish machine-readable author/plan schemas, serialization APIs, full custom-registration schemas, and exact Python graph helpers. The public sequence signatures and unary registration contract are fixed above; the JSON example is proposed encoding.
- Complete every built-in argument's type, automatic eligibility, scalar/pair normalization, and validation bounds beyond the defaults fixed in §6. In particular, fix normalization/leaky-ReLU defaults and concat axis normalization. Do not inherit changing backend defaults implicitly.
- Finalize DSL lexical rules, remaining positional mappings, generated-ID convention, and how structured plan restoration supplies the explicit backend registry. General graph helper APIs do not replace the primary DSL workflow.
- Define the supported representation for sharing non-batch dimension variables during resolution. Only batch remains symbolic in a successful plan; no general expression language is implied.
- Finalize canonical JSON encoding, semantic/artifact digest payloads, and golden test vectors.
- Fix initialization override and trainability syntax, build-receipt representation, runtime-check controls, and supported PyTorch versions/devices with numerical tolerances.
- Set configurable resolution/build limits, termination enforcement for custom rules, and the remaining diagnostic codes.

These are implementation interface decisions. They do not permit relaxing literal constraints, hiding architecture choices, or treating unresolved shapes as a successful plan.
