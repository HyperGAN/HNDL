# HNDL v1 technical specification

**Status: proposed; no implementation exists yet.** This document defines the baseline for implementing **HNDL — Human-readable Network Definition Language**, pronounced “handle.” It turns the direction in [DESIGN.md](DESIGN.md) into a standalone technical contract. [README.md](README.md) introduces the same ideas with examples.

“Must” denotes a v1 requirement. Python signatures and JSON examples describe the proposed interface, not an available package. This specification supersedes conflicting public API descriptions in DESIGN.md: the public frontends are a declarative subset of Python syntax and a separately invoked trusted Python callable. Both use explicit tensor arguments and share one graph/resolver. Omitted inferable dimensions replace explicit unknown markers; `name=` alone pins module identity; printing a module shows its resolved shapes. Open interface decisions are listed at the end; implementation must settle those before dependent features or serialized formats ship.

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

The initial deliverable includes declarative Python configuration and trusted callable capture for sequences and static named graphs, bounded shape inference, a PyTorch backend, layer-by-layer inspection, and custom operator registration. A JSON-compatible graph representation underlies resolution and supports the minimal custom multi-input extension. Object-based layer helpers are implementation details, not the primary public interface.

HNDL owns tensor contracts, architecture resolution, construction, module registration, plan persistence, and diagnostics. The host owns data semantics, losses, metrics, regularizers, optimizers, training schedules, runtime random streams, devices, and complete checkpoint recovery. Researchers can vary architecture specifications while keeping their metric evaluation loop unchanged; HNDL does not optimize a metric or search architectures automatically.

Topology search, dynamic image sizes, runtime control flow, weight tying between graph nodes, distributed training, export backends, a visual editor, and full StyleGAN recipes are outside v1. HNDL has no HyperGAN runtime dependency.

## 2. Core invariants

1. **Literal values are constraints.** Resolution must never replace a specified width, stride, kernel, activation, or binding to make a network fit.
2. **Inference is not selection.** Uniquely determined values may be inferred. Choosing among valid architectures requires a named policy or more explicit information.
3. **Resolution is pure.** Declarative configuration interpretation and built-in validation, resolution, and serialization require no torch import, tensor allocation, CUDA, random sampling, or network access. Trusted callable capture executes user Python before this pure resolution boundary and has no such side-effect guarantee.
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

Both single-input frontend APIs accept `input_shape` and `output_shape`, both including batch. Rank two means `BF`; rank four means `NCHW`. Unsupported ranks fail rather than guessing a layout. The `dtype` keyword defaults to `"float32"`. Structured graph contracts retain explicit layouts.

Omitting an inferable dimension creates a fresh unknown. For example, `linear(x)` leaves output width to resolution, while `linear(x, 64)` fixes it at 64. The operator schema identifies inferable dimension fields; omission does not make every argument inferable. Structured JSON likewise omits such fields rather than storing a special sentinel. No public unknown-value helper is required.

| Author value | Meaning |
| --- | --- |
| Literal | A hard constraint; a mismatch fails |
| Omitted inferable dimension | Resolve through supported relations or an eligible selected policy; otherwise fail |
| Omitted optional non-dimension field | Use its versioned operator default, unless reserved for an applicable policy |
| Omitted required non-inferable field | Fail, unless a selected policy explicitly supplies that field |

An unresolved dimension must not quietly become an operator default when inference stalls. Derived input widths/channels are operator relations, not additional required user arguments. `null`, `None`, `-1`, zero, arbitrary expressions, and keyword placeholders are not substitutes for omission. Each missing dimension is independent until a supported relation connects it.

HNDL does not infer color space, value range, activation choice, layout conversion, mixed precision, or broadcasting. A declared `tanh` is an explicit operation; an image output contract does not imply it. Joins must not insert resizing or broadcasting.

## 4. Author specification and graph

The canonical internal model is a finite directed acyclic graph. Both frontends produce this graph before resolution. Structured graph input remains available for named multi-input/multi-output networks.

| Field | Contract |
| --- | --- |
| `schema_version` | Author data-format version; initially `1` |
| `inputs` | Named external `TensorSpec` contracts |
| `nodes` | Ordered node declarations with stable `id`, versioned `op`, `args`, and input-port `inputs` bindings |
| `outputs` | Named output references with required tensor contracts |
| `policies` | Exact policy IDs/versions, arguments, and explicit node selections |
| `metadata` | Optional annotations excluded from numerical identity |

Input, output, and node IDs match `[a-z][a-z0-9_]*`. References are `input:<name>` or `node:<id>/<port>`. Built-in unary nodes consume `x` and produce `out`. Custom operators declare every input and output port.

The following proposed internal/structured JSON encoding defines a two-layer perceptron whose last width is resolved from the output contract. Ordinary users use either Python frontend in §7 instead. Container/key choices shown here are the baseline for the implementation; a machine-readable schema is still required before release.

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

Both frontends lower to external input `x` and one public output `output`. Declarative configuration selects its final `out` binding; a native callable returns its symbolic output. Every operator call creates one node in capture order. Explicit `name=` selects its stable node ID. Otherwise the ID is `n0`, `n1`, and so on, using the zero-based ordinal of that call among all operator calls, including explicitly named calls. Nested calls create their nodes in Python argument evaluation order. Duplicate explicit/generated IDs fail. Comments, local variable names, and assignment statements without new calls do not affect IDs. Inserting an operator call can change subsequent generated IDs; stable names are recommended for persisted models. Local variables refer to tensors and never infer node names.

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

The required initial policy is `spatial.up2_transpose@1`, selected as `policy="up2"` through the built-in registry alias:

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

The built-in operator alias `split` binds `split@1`. It takes input port `x`, output ports `first` then `rest`, an inferable `size`, and `dim` defaulting to `1`. It creates exactly two sections: the first `size` elements on `dim`, then all remaining elements. It does not repeatedly chunk the tensor at that size.

For input extent `I` on the selected axis and first section extent `s`, the relations are `first.dim = s`, `rest.dim = I - s`, and equivalently `I = s + rest.dim`. Require integers `1 <= dim < rank` and `0 < s < I`; batch splitting, negative axes, and empty sections fail. Every other dimension, layout, dtype, and batch value is equal across input and outputs. Omitted size is inferred from connected constraints when uniquely determined; otherwise ambiguity remains. No default chooses a section size.

```text
z1, z2 = split(x, 64)  # [B,128] gives [B,64] and [B,64]
```

For input `[B,160]`, the same statement yields `[B,64]` and `[B,96]`. The backend uses explicit sizes `[s, I-s]` in [`torch.split`](https://docs.pytorch.org/docs/stable/generated/torch.split.html) or equivalent views, preserving autograd and avoiding silent copies. The symbolic call returns a two-tensor tuple in declared output order; unpacking must match that arity. An unused output port is allowed, provided the split node remains reachable from the selected output.

## 7. Python frontends and public API

HNDL supports two explicit authoring modes. A string is parsed and interpreted as declarative configuration. A callable is executed only through the separately named trusted capture API. Neither mode falls back to the other.

### Declarative Python configuration

```python
from hndl.torch import network

source = """
hidden = linear(x, 64, name="hidden")
hidden = relu(hidden, name="activation")
out = linear(hidden, name="scores")
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

The external symbolic input is prebound as `x`; the final value bound to `out` selects the sole output. Every input tensor is passed explicitly. There is no implicit current tensor, assignment-derived module name, or separate output-selector argument.

`print(model)` uses the module's `__repr__`, following the standard PyTorch inspection convention, to display network input/output shapes and dtype followed by every layer's index, name, operation, and resolved input/output shapes. `repr(model)` returns the same representation. Inspection performs no forward pass, tensor allocation, or random draws. Illustrative output:

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name        operation  input shape  output shape
0      hidden      linear     [B, 128]     [B, 64]
1      activation  relu       [B, 64]      [B, 64]
2      scores      linear     [B, 64]      [B, 10]
```

### Trusted Python callable

```python
from hndl import ops
from hndl.torch import network_from_callable


def classifier(x):
    h = ops.linear(x, 64, name="hidden")
    h = ops.relu(h, name="activation")
    return ops.linear(h, name="scores")


model = network_from_callable(
    classifier,
    input_shape=("B", 128),
    output_shape=("B", 10),
    device="cpu",
)
```

`from hndl import ops` exposes built-in symbolic operators with fixed version bindings. `registry.ops.<alias>` exposes operators bound in an explicit registry, including custom operations. These calls capture graph nodes, not numerical torch operations. Native Python variables follow ordinary Python rules and never name modules. The callable is invoked exactly once with one symbolic input per API call and must return one symbolic tensor from that capture. A symbolic tensor cannot be used as a Python truth value or substituted with an eager tensor. Ordinary Python helpers and loops may construct a finite static graph; they do not become runtime graph control flow.

Symbolic factories are torch-free. Capture context is scoped to that call, isolated across concurrent captures, and cleaned up after success or exceptions; it is not a mutable global registry. Symbols from another capture fail. Coercion to an integer, boolean, iterable tensor values, or raw eager torch execution is unsupported and must fail rather than sample concrete values. Iterating a declared tuple of output symbols is ordinary tuple handling, not iteration over a symbolic tensor. The function is never invoked during forward execution, printing, serialization, or restoration; function source inspection is unnecessary.

Callable capture is explicitly trusted Python execution. The function and registered providers may have side effects, allocate tensors, import packages, or use randomness; HNDL does not sandbox them or claim that capture is pure. Hosts must choose trusted code accordingly. After capture, the shared resolver is pure. Capture errors propagate without retrying, executing source text, or silently switching frontend. Limits on the resulting graph do not bound arbitrary Python execution; hosts needing isolation must provide it outside this trusted API.

### Signatures and shared behavior

| API | Result and requirements |
| --- | --- |
| `resolve(source: str, *, input_shape, output_shape, dtype="float32", registry=None)` | Interpret declarative configuration and return an immutable `ResolvedPlan`, without torch |
| `resolve_file(path, *, input_shape, output_shape, dtype="float32", registry=None)` | Bounded UTF-8 file loading followed by the same declarative resolution pipeline |
| `resolve_callable(fn, *, input_shape, output_shape, dtype="float32", registry=None)` | Invoke trusted capture once, then resolve the captured graph |
| `network(source: str, *, input_shape, output_shape, device, dtype="float32", registry=None, initialization_seed=None)` | Declarative resolution plus construction; tensor-returning `forward(x)` |
| `network_file(path, *, input_shape, output_shape, device, dtype="float32", registry=None, initialization_seed=None)` | Bounded UTF-8 loading followed by the same declarative construction pipeline |
| `network_from_callable(fn, *, input_shape, output_shape, device, dtype="float32", registry=None, initialization_seed=None)` | Trusted capture, resolution, and construction with the same runtime interface |
| `model.plan` | Immutable resolved plan used to construct the module |
| `print(model)`, `repr(model)` | Complete layer/port shape table without sample tensors or execution |
| `print(plan)`, `repr(plan)` | Concise resolved shape representation, usable without torch |
| `plan.describe()` | Detailed shapes, provenance, asserted contracts, static parameter counts where known, and semantic digest |
| `hndl.torch.build(plan, *, device, initialization_seed=None)` | Lower-level construction returning a dictionary-output `GraphModule` |

`resolve`, `resolve_file`, `resolve_callable`, and `ops` are exported by `hndl`; construction functions are exported by `hndl.torch`. A string always means source, never a filename or function to execute. File paths are supplied only by the host through the explicit file APIs. A missing registry selects an independent built-in registry. Explicit registry objects supply extensions without a mutable global singleton. Built-in `ops` and explicit `registry.ops` calls capture exact operator identities; a capture using an identity unavailable in the selected resolution registry fails. Both modes use identical operator/policy rules and output contracts. Their graph has external input `x` and public output `output`; `output_shape` constrains the selected tensor. A configuration without a final symbolic `out`, or a callable returning anything except one symbolic tensor, fails. Unused output ports are allowed, but every created node must reach the selected output.

Both network constructors accept one runtime tensor and return one tensor, even for branched graphs. Printed graphs include all input/output ports, including both split results; a shared producer appears once. The lower-level graph builder retains named-input/dictionary-output behavior for structured graphs.

### Declarative grammar and trust boundary

The declarative frontend has version `python_config@1`, recorded with source provenance. It interprets a pinned, allowlisted Python AST subset. The accepted grammar is the following contract, not whatever a newer Python parser happens to accept:

- The module contains only simple assignment statements with one target. Targets are a local name or a flat tuple/list of distinct names for unpacking. Chained, annotated, augmented, attribute, subscript, and starred assignments fail.
- Assignment values are existing symbolic tensor names, flat tuples of symbolic tensors, or registered operator calls returning those values. Nested registered calls are permitted in tensor arguments. Literal-only local variables are not part of v1; literals belong in call arguments.
- Calls name a registered operator alias directly, such as `linear(...)`. Attribute calls such as `ops.linear(...)` are reserved for trusted native Python and fail in configuration. Calls cannot be redirected through local aliases.
- Tensor ports take symbolic tensors, positionally in declared port order or through their declared keyword names. Literal scalar arguments follow tensor ports according to the operator schema. Duplicate positional/keyword bindings and missing ports fail.
- Argument literals are bounded integers, finite floats, booleans, strings, and schema-permitted `None`, plus recursively bounded literal lists, tuples, and dictionaries with unique string keys. Positive/negative numeric literal signs are allowed; arithmetic and computed expressions are not. Containers cannot conceal calls or arbitrary objects. Strings are accepted only where the schema permits them; booleans are not valid integer dimensions. Inferable dimensions are omitted, not written as `None`.
- `name="hidden"` is reserved node metadata and `policy="up2"` selects a registered policy alias. Neither reaches an operator constructor as an ordinary argument. Schemas with colliding port/argument names must use nonconflicting names or structured data.
- There are no imports, function/class definitions, conditionals, loops, comprehensions, lambdas, arbitrary expressions, arbitrary function calls, attribute/subscript access, decorators, formatted strings, or star/keyword expansion. Expression statements, including standalone calls/docstrings, fail.

Python comments, whitespace, parentheses, and multiline calls follow the pinned grammar and preserve source locations for diagnostics. They do not change graph identity. Local names use ordinary non-keyword Python identifiers; special double-underscore names are rejected. Rebinding locals, including `x` and `out`, is allowed: evaluate the right-hand side against prior bindings, then bind the target. References to earlier tensors remain valid through other locals; rebinding does not mutate a tensor or rename/rebuild a node. Undefined and forward local references fail. Registered operator aliases cannot be rebound. The registry must reject aliases reserved for external `x` or output `out` in this frontend. The final `out` must hold one symbolic tensor, even if an earlier binding held a tuple.

Tuple results may be stored in a local and later unpacked; unpacking must match the declared number of outputs exactly. For example, `parts = split(x, 64)` followed by `z1, z2 = parts` is valid. Tensor variables and module IDs are distinct: `h = linear(x, 64)` creates generated node `n0`, while `hidden = linear(x, 64, name="hidden")` creates node `hidden`. Renaming `h` cannot change the state keys. No assignment introspection is performed in either frontend.

### Loading limits and trust boundaries

String and file entry points share one pipeline. Apply byte/line bounds before parsing; files are read as bounded UTF-8 with explicit errors for invalid encoding or oversized content, without reading an unbounded file first. Apply common-indentation removal compatible with `textwrap.dedent` so indented triple-quoted examples work. Keep a source map from dedented columns to original line/column positions. Files may not request other files, imports, or callbacks. The host supplies a path; configuration cannot choose what to load.

The interpreter must never use Python `eval`, `exec`, or compiled-code execution on configuration. Restricting `__builtins__` around `exec` is not a security boundary, as the [Python execution documentation](https://docs.python.org/3/library/functions.html#exec) explains. Parse to an AST, validate the complete allowlist and budgets before interpreting it, then dispatch only through the explicit registry. Literal values are interpreted directly with bounded traversal, not passed to an unbounded evaluator. Pin the supported parser grammar/version with the frontend and reject unsupported syntax even if the host interpreter accepts it. Publish compatibility fixtures before release. No configuration-supplied callback, import path, class, or provider may be loaded.

Enforce bounds on AST node count/depth, literal/collection sizes, graph nodes/edges, and resolution work. Parsing itself can exhaust resources before AST validation, as documented for [Python AST parsing and literal evaluation](https://docs.python.org/3/library/ast.html). All declarative source, including file content, follows the same isolated parser/validator worker path with memory/time limits by default. Bound the worker's returned data and diagnostics too, and validate their format before accepting them. Do not infer that a string is trusted from its origin or silently fall back to an unbounded parser. A Python AST allowlist alone is not a denial-of-service sandbox. Registry rules/builders remain trusted code and must not be selected from arbitrary external imports by configuration. Use bounded, explicitly declared argument schemas, not user callbacks for interpreting values. Trusted native capture remains the separately requested execution mode, not an alternate parser behavior.

Before module allocation, enforce dimension-product and total registered parameter/buffer byte limits for the concrete plan, using conservative upper bounds where exact counts are unavailable. Report static parameter counts where calculable. Custom builders must supply declared allocation bounds to participate in bounded construction; missing bounds fail when those limits cannot be established. A symbolic batch dimension prevents a complete runtime activation-memory estimate. Static bounds must not be presented as a guarantee of total training memory, and trusted custom implementations remain responsible for honoring their declarations.

### Operator calls and exact identities

Representative symbolic call signatures are `linear(x, [out_features], ...)`, `conv(x, [out_channels], ...)`, `deconv(x, [out_channels], ...)`, `reshape(x, *dimension_prefix)`, `split(x, [size], dim=1)`, and `add(a, b)`. Brackets denote an optional argument in this description, not literal syntax; `reshape(x, 512, 4, 4)` supplies literal variadic dimensions without using star expansion in source. Calls such as `relu(x)` preserve shape. Complete schemas determine other positional/keyword arguments.

A reshape prefix fixes the leading non-batch axes and bounds admissible output ranks. Length `K` requires at least `K` non-batch axes. `reshape(x, 512)` can describe BF or NCHW until connected rules narrow it; `reshape(x, 512, 4, 4)` uniquely establishes NCHW among supported layouts. Structured `shape` arrays use the same prefix semantics. `reshape(x)` is valid when connected rules fix rank and dimensions; `[B,2,4,4] → reshape → linear` gives BF width 32. Ambiguous rank/factorization fails rather than selecting arbitrary dimensions.

Configuration calls and `ops` attributes contain no version suffixes. Registry aliases bind immutable exact identities: `linear` → `linear@1`, `conv` → `conv2d@1`, `deconv` → `conv_transpose2d@1`, `split` → `split@1`, and policy `up2` → `spatial.up2_transpose@1`. Duplicate/conflicting registrations fail. Resolution never selects a newest installed version. Plans preserve exact identities and expanded policy effects, so restore does not depend on current alias bindings.

### Branches and reusable subnetworks

This configuration takes `[B,128]` and returns `[B,128]`:

```python
source = """
# Every input is explicit; local variables refer to tensors.
z1, z2 = split(x, 64, name="partition")
content = linear(z1, 128, name="content")
features = relu(content, name="features")
style = linear(z2, 128, name="style")
out = add(features, style, name="combined")
"""
```

A registered adaptive normalization fixture can use the remainder directly:

```python
source = """
z1, z2 = split(x, 64)
project = linear(z1)
features = reshape(project, 32, 4, 4)
out = adaptive_norm(features, z2)
"""
```

Input `[B,128]` and output `[B,32,4,4]` infer projection width 512, with the 64-feature remainder matching `2 * 32` style parameters. Custom aliases must be registered before interpretation. In native code the equivalent call is `registry.ops.adaptive_norm(features, z2)`.

A reusable subnetwork is an ordinary module registered under an alias such as `style_mlp`, with declared ports, shape relations, and backend construction. Configuration calls `style = style_mlp(z2)`; native code calls `style = registry.ops.style_mlp(z2)`. Each invocation creates an independent node/module. Trusted native helper functions may themselves compose symbolic ops; configuration does not define or call arbitrary Python helpers.

### Layer access

A lowered consecutive unary chain provides indexing, negative indexing, name lookup, length, and iteration over the actual registered layers:

```python
first = model[0]
last = model[-1]
hidden = model["hidden"]
assert first is hidden
assert len(model) == 3
layers = list(model)
features = model[:2]
```

Module lookup uses explicit or generated node IDs, never local variable names. Unknown IDs raise `KeyError`; out-of-range integer indexes raise `IndexError`. A slice returns ordinary `torch.nn.Sequential` over the existing module objects, sharing parameters, buffers, mode, and device moves. It carries no resolved-plan guarantees. Slicing requires adapters compatible with direct tensor-in/tensor-out chaining; otherwise reject it explicitly. Creating a slice does not alter registrations/state keys on the original model.

Registration remains once under `nodes.n_<node_id>`, without a facade prefix or duplicate registrations. Structural replacement/insertion/deletion through the HNDL container is rejected; ordinary parameter updates remain supported. Architecture changes require a new resolution/build. Actual split/fan-out/join graphs support node-name lookup only, regardless of which frontend authored them. Both network constructors still return the single selected tensor at runtime; structured `build` returns a dictionary-output GraphModule.

### Worked generator resolution

```python
from hndl.torch import network

source = """
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
model = network(
    source,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
print(model)
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

The last convolution preserves spatial size. Three doubling stages give `output_height = 8 * seed_height` and the same for width; projection width is `512 * seed_height * seed_width`.

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

`Registry.builtins()` returns an independently extensible registry containing the fixed built-in aliases. `register(alias, *, identity, version, shape, max_state_bytes, state_version=1)` binds a new alias to its exact operator identity/version. The no-argument unary helper shown here supplies an empty argument schema, input port `x`, output port `out`, default state compatibility version `1`, and unqualified capability status. State compatibility metadata belongs to the pure registration, so resolution can record it without loading a backend. More complex operators must declare their schemas/ports explicitly through the full extension API.

`max_state_bytes` is an explicit nonnegative per-node upper bound on registered parameter and buffer storage, including nonpersistent buffers, recorded in the pure registration and plan. The SiLU example declares zero because it owns neither parameters nor buffers; shape preservation alone does not imply this. Full providers may supply a pure bound derived from resolved arguments. The planner sums bounds before building, and the backend checks registered storage against the declaration after construction. These are trusted implementation contracts, not a sandbox for builder code or a bound on temporary allocations.

`preserves_shape` contributes bidirectional equality relations for input/output dimensions, layout, and dtype; it is not merely a forward shape callback. Downstream constraints can therefore propagate through this custom layer. The provider's claim must still be verified against the actual module during numerical qualification; registering it does not qualify its device/dtype/gradient behavior.

`register_torch(registry, alias, *, module, state_version)` attaches a backend constructor to the alias's already registered exact identity/version. Missing bindings, duplicate backend registration, and a state version differing from the pure declaration fail. In the no-argument unary form, the builder constructs `module()` once per node, invokes it with the `x` tensor, and binds its tensor result to `out`. There is no constructor introspection or assumption about other module arguments. `nn.SiLU` here uses its ordinary non-in-place constructor behavior.

Pure registration and `resolve()` remain usable without importing torch, even though the combined example imports torch to register a backend. Resolution must not call the backend constructor. Backend bindings stay separate from serializable arguments; neither configuration nor saved data can import a module class. The plan stores `example.silu@1` and its state compatibility requirement, not executable Python. Building without its explicitly registered backend must fail.

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

- Author/plan schema and resolution-semantics versions; exact operator and policy versions; frontend/grammar provenance for configuration or callable capture.
- Concrete nodes, edges, ports, arguments, tensor contracts, and verified/asserted status.
- Expanded initialization/trainability settings and stable execution order.
- Resolution trace, source locations, and value provenance.
- Semantic/artifact digests, implementation/dependency provenance, and state compatibility requirements.

Serialization uses canonical JSON with sorted object keys, ordered node lists, normalized tuple/numeric representations, no non-finite values, and SHA-256. Exact encoding and digest test vectors must be published before serialized plans ship.

The **semantic digest** covers concrete graph structure, state IDs, operator versions, arguments, contracts, initialization, trainability, and execution order. It excludes comments, source locations, and inference explanations. Policy provenance is preserved, but after effects are expanded, equivalent concrete numerical plans must have identical semantic identity regardless of authoring syntax or derivation.

The **artifact digest** covers the complete saved plan except its own digest field. It includes provenance and trace information that can differ between numerically equivalent plans. Parameter counts, where statically available, are not estimates of peak training memory.

## 10. PyTorch backend contract

`build(plan, *, device, initialization_seed=None)` returns a normal training-mode `GraphModule`. The network constructors present the single-input/single-output facade from §7 over the same construction semantics and registered state, with sequence indexing only for unary chains. Device is explicit; CPU is suitable for small fixtures and CUDA is the intended primary execution target. There is no silent CPU fallback. The plan fixes the qualified dtype, initially `float32`.

- A `GraphModule` validates declared external input contracts in `forward(**inputs)` and returns a dictionary keyed by public output names, including for a single output. Both network facades instead accept `forward(x)` and return the selected output tensor, preserving the same contract checks even for a branched graph.
- Modules register once under `nodes.n_<node_id>`; the prefix avoids collisions with module attribute names. Stateless nodes keep execution/diagnostic identities without state entries.
- Nodes run in stable topological order, breaking ties by declaration order. The order is saved in the plan.
- Parameters/buffers are fully materialized before optimizer or distributed setup. No first-forward parameter creation is allowed.
- Runtime must preserve ordinary gradients. It must not silently detach, clone, cast, move tensors, or mutate a shared branch through in-place built-in activations.
- Contract mismatches must be reported before applying the affected layer where possible. Asserted custom contracts require runtime validation.

Initialization rules belong to the plan; the build receipt records the selected seed mode and runtime. Initial support is versioned `torch_default@1` plus explicit constant overrides. Both network constructors and `build` default to `initialization_seed=None`: construction uses and advances the caller's ordinary PyTorch RNG state, so `torch.manual_seed(...)` controls initialization as it does for handwritten modules. Supplying an explicit integer instead uses an isolated RNG scope on the requested device and restores the caller's RNG state afterward. No seed argument is needed for ordinary use.

The supported PyTorch runtime must be recorded because its defaults affect initialization. Module construction order is stable; custom builders must respect the chosen RNG mode. Exact initial tensors across devices or PyTorch versions are not promised. Initialization draws happen at construction only; this does not permit hidden random draws during forward execution or representation/inspection.

An explicit initialization seed scopes module construction only. It neither isolates nor controls randomness or other effects inside an author callable, which runs earlier during trusted capture.

Per-node trainability masks are applied during construction and saved in the plan. Freezing parameters does not select evaluation mode. The host controls train/eval behavior and must preserve these masks. Stateful custom modules declare persistent buffers and manage derived caches correctly after state loading/device moves.

## 11. Persistence and compatibility

Persist the captured author graph and full resolved plan, plus declarative source or trusted callable provenance as applicable. Do not pickle a callable or closure as the architecture. Restoration must never execute the author callable or configuration to reconstruct saved state. Reconstruct from the saved plan, not by applying a newer resolver to old omitted dimension fields. Before state loading, validate the semantic digest, required implementation/state versions, and state names/shapes/dtypes.

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
| `E_NAME` | Undefined/forward local reference or attempted rebinding of a registered operator alias |
| `E_OUTPUT_ARITY` | Tuple unpacking has the wrong number of targets |
| `E_OUTPUT` | Final configuration `out` or native return value is missing or is not one tensor symbol |
| `E_SYNTAX` | Configuration contains an AST form outside `python_config@1` |
| `E_CAPTURE` | Native code mixes capture contexts or coerces a symbolic tensor into an eager value |
| `E_RESOURCE` | Source, parser-worker, AST, graph, or resolution budget is exceeded |
| `E_STATE_VERSION` | Saved state requires an unavailable compatible operator implementation |

Schema/version/resource-limit diagnostics also require stable codes; their complete catalog is pending. Configuration failures report original source line/column locations through the dedent map. Native capture locations are best effort and may use call-site information, but must always identify the affected node/field without requiring function-source inspection. `print(model)` and `repr(model)` must include every layer, its ID/operator, and complete input/output shapes without executing the network. Pure `print(plan)` exposes a concise shape representation without a backend. `plan.describe()` must additionally expose provenance sufficiently to explain why a field changed between separately resolved specifications.

Static resolution, dry-run/meta checks, and numerical preflight are distinct inspection modes. Numerical preflight must use disposable modules or restore affected state/RNG; it must not advance a live training stream or alter persistent buffers.

## 13. Acceptance criteria and implementation order

The following are future gates, not claims about tests already passing:

| Gate | Required evidence |
| --- | --- |
| Pure core | Interpret bounded configuration, resolve captured graphs, and serialize without torch/CUDA; malformed specs and unavailable providers fail clearly |
| Shape resolution | Omitted output widths/channels and reshape suffixes; rank resolution from prefix bounds and connected contracts, including bare reshape using upstream element count; ambiguous rank/factorization rejected; generator targets `32×32`, `64×64`, `32×64`; invalid `30×30`; literal-width conflict; inverse ambiguity; group divisibility and join conflicts |
| Determinism | Repeated resolution of the same captured graph is identical; equivalent configuration/native/structured graphs with identical node identities and declarations have identical semantic plans; exact alias bindings and policy-order independence |
| Configuration | AST allowlist rejects executable Python outside the subset; locals rebind without mutating edges; unpacking evaluates RHS once; nested calls follow evaluation order; names/comments do not rename nodes; `name` metadata does; final `out` required |
| Loading boundary | Bounded UTF-8 file reads, dedent/source mapping, isolated parse/validation limits, bounded literals, no eval/exec/import dispatch, no fallback between string/file/callable modes |
| Native capture | Trusted fn called exactly once; return one current-capture symbol; helpers/static loops work; context cleanup/concurrency; cross-capture/coercion failures; no callback execution during forward/inspection/restore |
| Backend equivalence | Handwritten PyTorch comparison from identical state: forward values, input/parameter gradients, optimizer updates |
| Public sequence API | Tensor forward result; complete shape table from `print(model)` without allocation/RNG draws; integer/negative/name lookup and iteration; shared-module slices; stable state keys without double registration; structural assignment rejected |
| Graph authoring | All ports explicit; branches/joins execute each node once; old tensor references survive rebinding; selected output receives its contract; unused ports allowed but dead nodes fail; node lookup only for branched models |
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

1. Pure schema/types, fixed registry bindings, bounded declarative AST interpreter/file loading, symbolic factories, linear/reshape/activation relations, trace, and plan serialization.
2. Explicit trusted callable capture, split/concat/branch relations, convolution rules, doubling policy, complete shape inspection, and frontend equivalence/determinism checks.
3. PyTorch `network` facade and lower-level builder, shape representation, layer access/slicing, state registration, caller-controlled/isolated initialization modes, CPU equivalence, and CUDA checks.
4. Simple custom-layer registration, then named graph joins/fan-out, full custom providers, and the affine-normalization fixture.
5. Complete both Python frontend walkthroughs and acceptance gates; host adapters follow the standalone core.

A sequence-only first milestone is useful progress, not completion of the graph/custom-extension v1 contract. Host adapters follow the standalone core.

## 14. Decisions to settle before implementation depends on them

The baseline deliberately leaves these details visible:

- Publish machine-readable author/plan schemas, serialization APIs, full custom-registration schemas, and exact Python graph helpers. The public frontend signatures, local-binding semantics, and unary registration contract are fixed above; the JSON example is proposed encoding.
- Complete every built-in argument's type, inference eligibility, scalar/pair normalization, and validation bounds beyond the defaults fixed in §6. In particular, fix normalization/leaky-ReLU defaults and concat axis normalization. Do not inherit changing backend defaults implicitly.
- Publish the pinned Python parser grammar/runtime compatibility matrix, AST allowlist fixtures, parser-worker budgets, remaining operator call schemas, and explicit backend registry restoration API. Generated IDs and local rebinding semantics follow §7; graph helpers do not replace the public frontends.
- Define the supported representation for sharing non-batch dimension variables during resolution. Only batch remains symbolic in a successful plan; no general expression language is implied.
- Finalize canonical JSON encoding, semantic/artifact digest payloads, and golden test vectors.
- Fix initialization override and trainability syntax, build-receipt representation, runtime-check controls, and supported PyTorch versions/devices with numerical tolerances.
- Set configurable resolution/build limits, termination enforcement for custom rules, and the remaining diagnostic codes.

These are implementation interface decisions. They do not permit relaxing literal constraints, hiding architecture choices, or treating unresolved shapes as a successful plan.
