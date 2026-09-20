# HNDL v1 technical specification

**Status: proposed; no implementation exists yet.** This document defines the baseline for implementing **HNDL — Human-readable Network Definition Language**, pronounced “handle.” It turns the direction in [DESIGN.md](DESIGN.md) into a standalone technical contract. [README.md](README.md) introduces the same ideas with examples.

“Must” denotes a v1 requirement. Python signatures and JSON examples describe the proposed interface, not an available package. This specification supersedes conflicting public API descriptions in DESIGN.md: the primary authoring surface is a multiline DSL with operator statements, optional tensor assignments, and input/output constraints passed through Python. Omitted inferable dimensions replace explicit unknown markers; optional names use `name=...`; printing a module shows its resolved shapes. Open interface decisions are listed at the end; implementation must settle those before dependent features or serialized formats ship.

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

The initial deliverable includes a DSL for sequences and static named graphs, a typed Python construction API, bounded shape inference, a PyTorch backend, layer-by-layer inspection, and custom operator registration. A JSON-compatible graph representation underlies resolution and supports the minimal custom multi-input extension. Object-based layer helpers are implementation details, not the primary public interface.

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

## 3. Tensor contracts and omitted dimensions

A `TensorSpec` contains ordered dimensions including batch, an explicit layout, and a dtype. Initially supported layouts are:

| Layout | Dimensions | Meaning |
| --- | --- | --- |
| `BF` | `[B, F]` | Batch and feature width |
| `NCHW` | `[B, C, H, W]` | Batch, channels, height, width |

Qualified v1 execution uses `float32`. A shared batch symbol, written `"B"` in these examples, enforces equality across participating inputs and outputs. Concrete dimensions and runtime batch sizes must be positive integers. Only batch may remain symbolic after resolution; symbolic batch is preserved through reshape and flatten.

The single-input DSL API accepts `input_shape` and `output_shape`, both including batch. Rank two means `BF`; rank four means `NCHW`. Unsupported ranks fail rather than guessing a layout. The `dtype` keyword defaults to `"float32"`. Structured graph contracts retain explicit layouts.

Omitting an inferable dimension creates a fresh unknown. For example, `linear` leaves output width to resolution, while `linear 64` fixes it at 64. The operator schema identifies inferable dimension fields; omission does not make every argument inferable. Structured JSON likewise omits such fields rather than storing a special sentinel. No public unknown-value helper is required.

| Author value | Meaning |
| --- | --- |
| Literal | A hard constraint; a mismatch fails |
| Omitted inferable dimension | Resolve through supported relations or an eligible selected policy; otherwise fail |
| Omitted optional non-dimension field | Use its versioned operator default, unless reserved for an applicable policy |
| Omitted required non-inferable field | Fail, unless a selected policy explicitly supplies that field |

An unresolved dimension must not quietly become an operator default when inference stalls. Derived input widths/channels are operator relations, not additional required user arguments. `null`, `None`, `-1`, zero, arbitrary expressions, and keyword placeholders are not substitutes for omission. Each missing dimension is independent until a supported relation connects it.

HNDL does not infer color space, value range, activation choice, layout conversion, mixed precision, or broadcasting. A declared `tanh` is an explicit operation; an image output contract does not imply it. Joins must not insert resizing or broadcasting.

## 4. Author specification and graph

The canonical internal model is a finite directed acyclic graph. DSL statements lower into this graph before resolution. Structured graph input remains available for named multi-input/multi-output networks.

| Field | Contract |
| --- | --- |
| `schema_version` | Author data-format version; initially `1` |
| `inputs` | Named external `TensorSpec` contracts |
| `nodes` | Ordered node declarations with stable `id`, versioned `op`, `args`, and input-port `inputs` bindings |
| `outputs` | Named output references with required tensor contracts |
| `policies` | Exact policy IDs/versions, arguments, and explicit node selections |
| `metadata` | Optional annotations excluded from numerical identity |

Input, output, and node IDs match `[a-z][a-z0-9_]*`. References are `input:<name>` or `node:<id>/<port>`. Built-in unary nodes consume `x` and produce `out`. Custom operators declare every input and output port.

The following proposed internal/structured JSON encoding defines a two-layer perceptron whose last width is resolved from the output contract. Ordinary users write the DSL in §7 instead. Container/key choices shown here are the baseline for the implementation; a machine-readable schema is still required before release.

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
      "args": {"bias": true},
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

Validation must reject duplicate IDs, missing references, undeclared ports, missing required bindings, cycles, and nodes that cannot reach any declared output. Unused output ports of an otherwise reachable node are allowed. Multiple public outputs are allowed. Graph values are tensors; adapters must flatten Python tuple/dict results into declared tensor ports.

Each node owns an independent module instance and executes once per forward. Fan-out reuses a computed tensor. It does not clone the source module, repeat its invocation, or tie parameters between nodes. Explicit weight tying between graph nodes is deferred and must fail if requested. External reuse of built modules through sequence slices is supported as described in §7.

The DSL lowers to external input `x` and one public output `output`, selected as described in §7. Simple implicit connections produce a unary chain; assignments and explicit port bindings can produce a static graph. Structured authoring requires explicit node IDs. The DSL assigns deterministic position-based IDs to unnamed statements, shows them in the printed table and expansion, and warns that inserting operations changes later generated names. Comments and blank lines do not count as statements. Explicit names and single-output assignment names become stable node IDs; collisions between explicit and generated IDs fail. Tensor aliases and module/node IDs occupy distinct namespaces, with the single-output naming equivalence defined in §7.

## 5. Resolution algorithm

The resolver must perform the following logical steps:

1. Validate the schema and graph; bind exact operator and policy versions from the active registry.
2. Expand explicitly declared finite policy templates, if any, and validate their nodes/edges. The required initial policy does not change topology.
3. Seed tensor contracts and literals. Apply policy requirements as constraints. Create fresh unknowns for omitted inferable dimensions and add operator relations. Apply operator defaults only to other omitted fields not already supplied by policy requirements or reserved for selected policy defaults.
4. Propagate forward and backward to a fixed point, recording derivations and detecting contradictions.
5. Ask selected policies to resolve remaining eligible choices according to their deterministic rules, then propagate again.
6. Fail if any required construction value remains ambiguous or unresolved; otherwise freeze the concrete plan.

Rules may only add or refine facts. They must not retract constraints or overwrite literals. The resolver must bound expansion, dimension products, candidate sets, and custom-rule execution. Exceeding a configured limit is an explicit failure, never silent truncation. Exact limits/configuration APIs remain an implementation decision.

The required relation vocabulary includes positive integers, equality, integer sums/differences for split and concat, multiplication by a constant, products, divisibility, and fixed convolution equations. Direct equations, finite candidates, and bounded interval reasoning are sufficient; a general symbolic or SMT solver is not required.

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

It supplies omitted eligible values and rejects contrary literals. These settings guarantee exact spatial doubling; channel widths and bias are separate arguments. The policy does not guarantee image quality. It must not silently substitute resize/convolution, crop, or a different architecture when a target fails.

Custom policies are trusted, explicitly registered Python providers. Data files cannot contain executable expressions or arbitrary imports.

## 6. Built-in operators

The minimum catalog is listed below; each identifier has semantic version `@1`. Unary operators use ports `x`/`out` unless stated otherwise. All propagate batch equality and preserve dtype. Equations describe shape relations, not a license to accept arguments that the backend rejects.

| Operator | Arguments/ports relevant to shape | Required relation |
| --- | --- | --- |
| `linear` | `out_features`, optional bias | `BF → BF`; infer input width, constrain output width; no implicit flatten |
| `reshape` | Prefix of non-batch `shape` | Preserve per-example element count and batch; output must have a supported layout |
| `flatten` | No shape choices | Combine all non-batch dimensions into `BF` |
| `conv2d` | `out_channels`, kernel, stride, padding, dilation, groups | `NCHW → NCHW`; use the formula below independently per spatial axis |
| `conv_transpose2d` | As above, plus `output_padding` | `NCHW → NCHW`; use the transpose formula below |
| `relu` | None affecting shape | Preserve dimensions/layout; out-of-place |
| `leaky_relu` | Negative slope | Preserve dimensions/layout; out-of-place |
| `tanh` | None affecting shape | Preserve dimensions/layout; out-of-place |
| `group_norm` | Explicit positive group count | Preserve shape; groups must divide channels |
| `add` | Ports `a`, `b`; output `out` | Shapes, layout, dtype, and batch must match exactly |
| `split` | Input `x`; ordered outputs `first`, `rest`; inferable `size`, `dim=1` | Exactly two positive sections along a non-batch axis; input extent equals the sum of section extents |
| `concat` | Declared input count `K`, axis, ports `x0` through `x(K-1)` | Sum the selected non-batch axis; all other axes/layout/dtype must agree |

For each spatial axis, with input extent `I`, output extent `O`, kernel `k`, stride `s`, padding `p`, and dilation `d`:

```text
conv2d:           O = floor((I + 2p - d(k - 1) - 1) / s + 1)
conv_transpose2d: O = (I - 1)s - 2p + d(k - 1) + output_padding + 1
```

Dimensions, channels, kernel sizes, strides, and dilations must be positive. Convolution padding/output padding must be nonnegative; group divisibility and backend argument restrictions must also hold. The complete versioned argument schema must publish the allowed scalar/pair forms, defaults, and inferable fields before these operators ship.

Defaults required by the examples are fixed for semantic version 1: `linear.bias=true`; convolution `bias=true`, `stride=1`, `padding=0`, `dilation=1`, and `groups=1`; transposed convolution additionally has `output_padding=0`. Convolution `kernel_size` is required unless a selected policy supplies it. Output channels and linear output widths are inferable when omitted; they have no fallback default. The `up2` policy's requirements take precedence over these defaults. Built-in activations are out-of-place. These are HNDL versioned choices, not dynamically inherited backend defaults.

Ordinary convolution inversion may produce an integer interval, not a unique input. Transposed convolution does not inherently mean doubling. `reshape` may directly solve exactly one unknown product factor; multiple unknown factors require independent relations or a selected policy. Each omitted dimension is a distinct unknown unless a supported relation connects it. For example, known target height and width may separately determine two omitted seed axes.

For concat, axes index the complete tensor shape and axis `0` is batch, which cannot be concatenated in v1. The spelling/normalization of negative axes remains to be fixed in the argument schema. No join introduces broadcasting, casts, or layout conversion.

### Split and remainder

The built-in DSL alias `split` binds `split@1`. It takes input port `x`, output ports `first` then `rest`, an inferable `size`, and `dim` defaulting to `1`. It creates exactly two sections: the first `size` elements on `dim`, then all remaining elements. It does not repeatedly chunk the tensor at that size.

For input extent `I` on the selected axis and first section extent `s`, the relations are `first.dim = s`, `rest.dim = I - s`, and equivalently `I = s + rest.dim`. Require integers `1 <= dim < rank` and `0 < s < I`; batch splitting, negative axes, and empty sections fail. Every other dimension, layout, dtype, and batch value is equal across input and outputs. Omitted size is inferred from connected constraints when uniquely determined; otherwise ambiguity remains. No default chooses a section size.

```text
z1, z2 = split 64  # [B,128] gives [B,64] and [B,64]
```

For input `[B,160]`, the same statement yields `[B,64]` and `[B,96]`. The backend uses explicit sizes `[s, I-s]` in [`torch.split`](https://docs.pytorch.org/docs/stable/generated/torch.split.html) or equivalent views, preserving autograd and avoiding silent copies. Output binding count must be exactly two. An unused output port is allowed, provided the split node remains reachable from the selected output.

## 7. DSL and public API

The primary API accepts a multiline string with one operation per nonblank statement line and tensor constraints supplied separately:

```python
from hndl.torch import network

source = """
linear 64 name=hidden
relu name=activation
linear name=scores
"""
model = network(
    source,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
print(model)
# With a caller-supplied tensor x of shape [B, 128]:
# scores = model(x)
```

`print(model)` uses the module's `__repr__`, following the standard PyTorch inspection convention, to display network input/output shapes and dtype followed by every layer's index, name, operation, and resolved input/output shapes. `repr(model)` returns the same representation. Inspection performs no forward pass, tensor allocation, or random draws. The following console output is illustrative; no implementation has produced it yet:

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
| `resolve(source, *, input_shape, output_shape, output=None, dtype="float32", registry=None)` | Parse and resolve a single-input/single-output DSL graph into an immutable `ResolvedPlan`, without torch |
| `network(source, *, input_shape, output_shape, device, output=None, initialization_seed=None, dtype="float32", registry=None)` | Resolve and build a module with tensor-returning `forward(x)`, for a sequence or branched graph |
| `model.plan` | The immutable resolved plan used to construct the module |
| `print(model)`, `repr(model)` | A complete layer input/output shape table through `__repr__`, with no sample tensors or execution |
| `print(plan)`, `repr(plan)` | A concise resolved shape representation, usable without torch |
| `plan.describe()` | Detailed node/port shapes, value provenance, asserted contracts, static parameter counts where known, and semantic digest |
| `hndl.torch.build(plan, *, device, initialization_seed=None)` | Lower-level construction returning a dictionary-output `GraphModule` |

`resolve` is exported by `hndl`; `network` is exported by `hndl.torch`. The latter is the resolve/build convenience API and must not introduce different resolution semantics. A missing registry selects an independent built-in registry; explicit registry objects provide custom extensions without a mutable global singleton. Structured graph resolution and serialization helpers remain to be named; their data contracts are defined here.

The optional `output` keyword is either `None` or a tensor-alias string. A string selects that previously defined alias; `None` selects the final implicit current tensor. `output_shape` constrains the selected tensor. A terminal multi-output operation leaves no current tensor, so it requires an explicit selector, for example `output="z2"` after a split. Undefined aliases or a missing current tensor fail. The selected tensor becomes public output `output` in the lowered graph. Other nodes must still reach that selected output; selecting an earlier alias does not silently discard later dead nodes.

`network(...)` always exposes one tensor input and one tensor result, whether its lowered graph is a chain or has branches. `print(model)` and `print(plan)` show every input and output port for graph nodes, including both outputs of a split; shared producers appear once.

### Statements, names, and exact version bindings

Each statement consists of an optional output assignment followed by an operator alias, positional arguments, and named fields:

```text
[output_alias [, output_alias ...] =] operator [arguments ...]
```

Brackets denote optional grammar, not literal source characters. Assignment may bind one output (`hidden = linear 64`) or several (`z1, z2 = split 64`). Without assignment, an operator-first line such as `linear 64 name=hidden` remains valid. Required spaces separate the operator from positional arguments.

A `#` starts a comment extending to the end of the physical line, including after a statement. Strip comments before parsing while retaining original line/column locations for errors. Blank and comment-only lines do not affect generated node IDs or semantic digests. The initial DSL has no quoted strings, escape syntax, multiline statements, or nested blocks.

Optional `name=<id>` is reserved frontend metadata selecting the node ID; `policy=<alias>` selects a registered construction policy. Neither is an operator constructor argument. For a single-output operation, `hidden = linear 64` and `linear 64 name=hidden` are equivalent: both create node `hidden` and bind tensor alias `hidden` to its sole declared output port (`out` for built-ins). Using both with the same name is redundant but valid; differing assignment and `name` values fail. A single-output operation without either gets a generated node ID but no user tensor alias.

For multiple outputs, the assignment aliases bind ports in the operator's declared output order. `z1, z2 = split 64 name=partition` creates node `partition` and aliases for its `first`/`rest` outputs. Without `name`, the split node gets a generated ID. Aliases `z1`/`z2` are not module IDs. Multi-output operations require an assignment whose arity exactly matches their declared outputs.

Aliases and node names match the ID rules in §4. Tensor aliases are immutable within one model's scope: external alias `x` is reserved, rebinding fails, and undefined or forward alias references fail. A node ID may coincide with a tensor alias in its separate namespace. Aliases identify graph edges, not tensors stored globally or persistently on the module; each forward creates its own runtime tensor values.

Input ports use explicit bindings such as `x=z2`, `a=features b=style`, or `x=features params=z2`. The registered operator schema separates input ports from scalar arguments. Port names, `name`, and `policy` must not silently become scalar constructor arguments; conflicting scalar-field declarations must use a nonconflicting schema or structured data. Input ports named `name` or `policy` also require structured input because those keys are reserved in the DSL. Structured input keeps `args` and input-port bindings separate.

An implicit current tensor starts as external `x`. A unary operation without an explicit binding consumes it. Every single-output operation, including a join, updates the current tensor to its output. A multi-output operation clears current; the following unary operation must therefore bind its input explicitly. All required ports of multi-input operations must always be explicit. Process statements in source order; earlier aliases remain available even as current changes.

Boolean literals are lowercase `true` and `false`. Allow integers, finite numeric values, and registered policy aliases. Unknown operators, policies, arguments, duplicate node names, generated-name collisions, duplicate/rebound aliases, and repeated argument/port assignments fail with source locations. Legacy `id:` prefixes and an explicit unknown keyword are not part of the grammar.

The initial positional mappings are `linear [out_features]`, `conv [out_channels]`, `deconv [out_channels]`, `split [size]`, and `reshape [non-batch dimension prefix...]`; brackets here mean optional syntax, not literal DSL characters. Omitted output widths/channels are inferred through connected contracts and rules. `relu`, `tanh`, and `flatten` require no positional arguments. Other scalar arguments use their operator schema's field names, such as `kernel_size=3`. Complex/tuple-valued custom arguments use structured data until separately specified. Multi-input joins use explicit named input-port bindings in the DSL or structured graph data.

For reshape, supplied dimensions fix the leading non-batch axes and constrain admissible output ranks. A prefix of length `K` requires at least `K` non-batch axes; it does not require exactly `K`. Resolve rank/layout from these bounds and connected contracts/operator rules over the supported layouts. Thus `reshape 512` permits either BF or NCHW until context narrows it: before deconvolution it fixes NCHW channels and leaves height/width to inference, while a connected BF contract makes it a complete feature width. `reshape 512 4 4` requires at least three non-batch axes, which uniquely establishes NCHW among the supported layouts and supplies its complete target shape.

Structured `shape` arrays use the same prefix semantics; omitting `shape` supplies no dimensions. A reshape with no dimensions is valid when connected constraints establish its rank/layout and uniquely determine its full output shape. For example, `[B,2,4,4] → reshape → linear 10` infers BF from the linear input contract and feature width 32 from the upstream element count. Reject prefixes too long for the established rank and conflicting literals. If multiple supported ranks remain, report ambiguity rather than assuming the prefix is complete. Once rank is known, each missing factor still requires enough relations to determine it; element count alone cannot choose a factorization.

The DSL does not use `@` version suffixes. Registry bindings map aliases to exact immutable identities: `linear` to `linear@1`, `conv` to `conv2d@1`, `deconv` to `conv_transpose2d@1`, and policy `up2` to `spatial.up2_transpose@1`. Other built-in aliases bind the matching catalog operator at version 1. Custom aliases use explicit registration (§8). Duplicate/conflicting aliases fail. Resolution must never choose a newest installed version or discover arbitrary plugins. Saved plans materialize exact identities/versions and the effects of policy selection, so restoration does not depend on current alias bindings.

There is no Python evaluation, arithmetic, loops, runtime control flow, nested subnetwork grammar, persistent tensor context, or imports in the DSL. Its semantics must match equivalent structured graph input. Exact lexical details beyond this baseline must be fixed in the parser schema before shipping.

### Branching and reusing tensors

This complete built-in example takes `[B,128]` and returns `[B,128]`:

```python
from hndl.torch import network

branches = network(
    """
# Divide the input into a content path and a style path.
z1, z2 = split 64 name=partition
content = linear 128 x=z1
features = relu
style = linear 128 x=z2
add a=features b=style name=combined
""",
    input_shape=("B", 128),
    output_shape=("B", 128),
    device="cpu",
)
print(branches)
```

The split clears current, so `content` explicitly selects `z1`. `features` implicitly consumes the new current tensor `content`. The style path explicitly selects the earlier `z2`; the final add names both inputs and becomes current. This is a branched model, so `branches["partition"]` is supported while sequence slicing is not.

The DSL is also sufficient to wire the registered adaptive normalization fixture from §8:

```text
z1, z2 = split 64
project = linear x=z1
features = reshape 32 4 4
adaptive_norm x=features params=z2
```

With input `[B,128]` and output `[B,32,4,4]`, projection width resolves to 512, the feature path has 32 channels, and the style remainder has the required `2 * 32 = 64` features. `adaptive_norm` must be explicitly registered; the DSL does not define its implementation. The split and projection each execute once, and both paths remain differentiable.

A reusable subnetwork can be an ordinary module registered under an operator alias such as `style_mlp`, with declared ports, arguments, shape relations, and backend construction. For example, `style = style_mlp x=z2` selects an existing branch and binds the subnetwork output for later use. It follows the same custom-operator contract, gets an independent module instance per node, and requires registration before parsing/resolution. There is no nested block or function-definition syntax in v1.

### Layer access

A model whose lowered graph is a consecutive unary chain provides sequence access to the actual registered PyTorch layers. Assignments, names, and comments alone do not remove this access:

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

Every selected module must be directly callable with one positional tensor and return one tensor. If a custom node requires an invocation adapter to translate keyword-only inputs or unpack tuple/dict results, slicing across it raises `TypeError` rather than bypassing that adapter. Named/integer lookup still returns its registered module; the complete HNDL model uses the declared adapter.

The facade must preserve the backend's single registration under `nodes.n_<node_id>` and corresponding state keys, without registering each layer a second time or adding a wrapper prefix. Parameter access, gradients, optimizers, state dictionaries, device moves, and train/eval use ordinary PyTorch behavior. Structural replacement, insertion, or deletion through the HNDL container is rejected to keep its plan sound; ordinary parameter value updates remain supported. Architecture changes require resolving/building a new model.

Models with actual split, fan-out, or join structure support node-name lookup only; positional indexing, slicing, and sequence iteration are not graph APIs. `network(...)` still accepts one tensor and returns the selected tensor for these models. The lower-level `GraphModule` returned by `build(...)` retains `forward(**inputs)` returning a dictionary of public named tensors.

### Worked generator resolution

The generator uses three explicit doubling stages to work backward from an output contract:

```python
from hndl.torch import network

source = """
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
model = network(
    source,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
print(model)
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
| Arguments | Types, required/default values, inference eligibility, validation limits |
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
| `project` | `linear@1`, output width omitted | `x=node:w/out` | `[B,1024]` |
| `seed` | `reshape@1`, `shape=[64,4,4]` | `x=node:project/out` | `[B,64,4,4]` |
| `style` | `linear@1`, output width omitted | `x=node:w/out` | `[B,128]` |
| `norm` | Custom `adaptive_norm@1` | `x=node:seed/out`, `params=node:style/out` | `[B,64,4,4]` |

Bind public output `features` to `node:norm/out`. Both branches consume the same `w` tensor; `mapping` and `w` each execute once. Reshape determines the projection width as `64 * 4 * 4`; the custom relation below determines the style width as `2 * 64`.

The fixture operator, referred to as `adaptive_norm@1` here, has ports `x`, `params`, and `out`:

```text
x:      [B, C, H, W]   NCHW
params: [B, 2*C]       BF
out:    [B, C, H, W]   NCHW
```

The relation `params.features = 2 * x.channels` must resolve an omitted style-affine output width. Batch/dtype equality and output-shape equality are required. Its exact numerical convention is:

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

`build(plan, *, device, initialization_seed=None)` returns a normal training-mode `GraphModule`. `network(...)` presents the single-input/single-output facade from §7 over the same construction semantics and registered state, with sequence indexing only for unary chains. Device is explicit; CPU is suitable for small fixtures and CUDA is the intended primary execution target. There is no silent CPU fallback. The plan fixes the qualified dtype, initially `float32`.

- A `GraphModule` validates declared external input contracts in `forward(**inputs)` and returns a dictionary keyed by public output names, including for a single output. The `network` facade instead accepts `forward(x)` and returns the selected output tensor, preserving the same contract checks even for a branched graph.
- Modules register once under `nodes.n_<node_id>`; the prefix avoids collisions with module attribute names. Stateless nodes keep execution/diagnostic identities without state entries.
- Nodes run in stable topological order, breaking ties by declaration order. The order is saved in the plan.
- Parameters/buffers are fully materialized before optimizer or distributed setup. No first-forward parameter creation is allowed.
- Runtime must preserve ordinary gradients. It must not silently detach, clone, cast, move tensors, or mutate a shared branch through in-place built-in activations.
- Contract mismatches must be reported before applying the affected layer where possible. Asserted custom contracts require runtime validation.

Initialization rules belong to the plan; the build receipt records the selected seed mode and runtime. Initial support is versioned `torch_default@1` plus explicit constant overrides. Both `network` and `build` default to `initialization_seed=None`: construction uses and advances the caller's ordinary PyTorch RNG state, so `torch.manual_seed(...)` controls initialization as it does for handwritten modules. Supplying an explicit integer instead uses an isolated RNG scope on the requested device and restores the caller's RNG state afterward. No seed argument is needed for ordinary use.

The supported PyTorch runtime must be recorded because its defaults affect initialization. Module construction order is stable; custom builders must respect the chosen RNG mode. Exact initial tensors across devices or PyTorch versions are not promised. Initialization draws happen at construction only; this does not permit hidden random draws during forward execution or representation/inspection.

Per-node trainability masks are applied during construction and saved in the plan. Freezing parameters does not select evaluation mode. The host controls train/eval behavior and must preserve these masks. Stateful custom modules declare persistent buffers and manage derived caches correctly after state loading/device moves.

## 11. Persistence and compatibility

Persist the author specification and full resolved plan. Reconstruct from the saved plan, not by applying a newer resolver to old omitted dimension fields. Before state loading, validate the semantic digest, required implementation/state versions, and state names/shapes/dtypes.

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
| `E_ALIAS` | Tensor alias is undefined, referenced before definition, or rebound; assignment name conflicts with `name` |
| `E_OUTPUT_ARITY` | Multi-output assignment has the wrong number of aliases or is missing |
| `E_CURRENT` | An implicit input or default output is requested after split cleared the current tensor |
| `E_STATE_VERSION` | Saved state requires an unavailable compatible operator implementation |

Schema/version/resource-limit diagnostics also require stable codes; their complete catalog is pending. DSL failures include line/column locations where applicable. `print(model)` and `repr(model)` must include every layer, its ID/operator, and complete input/output shapes without executing the network. Pure `print(plan)` exposes a concise shape representation without a backend. `plan.describe()` must additionally expose provenance sufficiently to explain why a field changed between separately resolved specifications.

Static resolution, dry-run/meta checks, and numerical preflight are distinct inspection modes. Numerical preflight must use disposable modules or restore affected state/RNG; it must not advance a live training stream or alter persistent buffers.

## 13. Acceptance criteria and implementation order

The following are future gates, not claims about tests already passing:

| Gate | Required evidence |
| --- | --- |
| Pure core | Import, parse DSL, resolve, and serialize with no torch/CUDA; malformed specs and unavailable providers fail clearly |
| Shape resolution | Omitted output widths/channels and reshape suffixes; rank resolution from prefix bounds and connected contracts, including bare reshape using upstream element count; ambiguous rank/factorization rejected; generator targets `32×32`, `64×64`, `32×64`; invalid `30×30`; literal-width conflict; inverse ambiguity; group divisibility and join conflicts |
| Determinism | Repeated resolution is identical; equivalent DSL/structured graph inputs have the same semantic plan; aliases bind exact versions; policy order has no effect |
| DSL | Single-output assignment and `name` equivalence; arity and alias immutability; comments/blank lines preserve IDs and semantic identity; reserved fields stay out of constructor arguments; undefined/forward references, name conflicts, and collisions fail |
| Backend equivalence | Handwritten PyTorch comparison from identical state: forward values, input/parameter gradients, optimizer updates |
| Public sequence API | Tensor forward result; complete shape table from `print(model)` without allocation/RNG draws; integer/negative/name lookup and iteration; shared-module slices; stable state keys without double registration; structural assignment rejected |
| Graph DSL | Split clears current; explicit branches/joins execute each node once; earlier aliases remain usable; selected output receives its contract; terminal split requires selector; unused ports allowed but dead nodes fail; node lookup only for branched models |
| Split | `128 → 64+64` and `160 → 64+96`; inverse size inference; omitted ambiguous size rejected; non-batch positive axis and nonempty sections validated; both outputs preserve autograd |
| Unary registration | `silu` example resolves through bidirectional shape equality; pure resolution never constructs/imports a backend; exact identity/state version is preserved |
| Custom graph | Resolve `2*C`; fan-out works; feature/style gradients are correct; reject incorrect batch/channel contracts |
| Numerical validity | Finite affine-normalization forward/backward for constant/nonconstant inputs; first/second derivative checks on suitable nondegenerate fixtures |
| Registration and recovery | Stable registered state; fresh-process round-trip; no first-forward parameters or hidden runtime RNG draws |
| Initialization | Default construction follows caller RNG and `torch.manual_seed`; explicit construction seeds isolate/restore RNG; inspection consumes no RNG |
| CUDA | Selected operators and custom fixture execute and reload on CUDA within declared tolerances |
| Compatibility | Load compatible saved plans without replanning; reject incompatible semantics/state versions actionably |

No identity substitutes, blanket skips, fabricated successful output, or silent device fallback can satisfy these gates. CPU correctness does not establish CUDA correctness. Host integration, complete checkpoint recovery, two-GPU behavior, and multi-host behavior are separate integration gates.

Implement in this order:

1. Pure schema/types, fixed registry aliases, DSL statements/comments/assignments and port lowering, linear/reshape/activation relations, trace, and plan serialization.
2. Split/concat relations and named branch/join semantics, convolution relations, doubling policy, complete shape inspection, and DSL/graph semantic equivalence and determinism checks.
3. PyTorch `network` facade and lower-level builder, shape representation, layer access/slicing, state registration, caller-controlled/isolated initialization modes, CPU equivalence, and CUDA checks.
4. Simple custom-layer registration, then named graph joins/fan-out, full custom providers, and the affine-normalization fixture.
5. Complete the public DSL walkthroughs and acceptance gates; host adapters follow the standalone core.

A sequence-only first milestone is useful progress, not completion of the graph/custom-extension v1 contract. Host adapters follow the standalone core.

## 14. Decisions to settle before implementation depends on them

The baseline deliberately leaves these details visible:

- Publish machine-readable author/plan schemas, serialization APIs, full custom-registration schemas, and exact Python graph helpers. The public DSL signatures, assignment/binding semantics, and unary registration contract are fixed above; the JSON example is proposed encoding.
- Complete every built-in argument's type, inference eligibility, scalar/pair normalization, and validation bounds beyond the defaults fixed in §6. In particular, fix normalization/leaky-ReLU defaults and concat axis normalization. Do not inherit changing backend defaults implicitly.
- Finalize DSL lexical rules, remaining positional mappings, generated-ID convention, and how structured plan restoration supplies the explicit backend registry. General graph helper APIs do not replace the primary DSL workflow.
- Define the supported representation for sharing non-batch dimension variables during resolution. Only batch remains symbolic in a successful plan; no general expression language is implied.
- Finalize canonical JSON encoding, semantic/artifact digest payloads, and golden test vectors.
- Fix initialization override and trainability syntax, build-receipt representation, runtime-check controls, and supported PyTorch versions/devices with numerical tolerances.
- Set configurable resolution/build limits, termination enforcement for custom rules, and the remaining diagnostic codes.

These are implementation interface decisions. They do not permit relaxing literal constraints, hiding architecture choices, or treating unresolved shapes as a successful plan.
