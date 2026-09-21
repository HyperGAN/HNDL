# HNDL v1 technical specification

**Status: the implemented 0.1.2 contract.** This document defines the technical contract for **HNDL — Human-readable Network Definition Language**, pronounced “handle.” [IMPLEMENTATION.md](IMPLEMENTATION.md) records what 0.1.2 ships and what remains for v1. [README.md](README.md) introduces the APIs with examples.

“Must” denotes a requirement of this contract. The public frontends are a declarative subset of Python syntax and a separately invoked trusted Python callable. Both share an implicit current tensor for single-input operations, accept explicit tensors for branches, and use one graph/resolver. Every operation — built-in or custom — is an `nn.Module` carrying one `@operator(...)` declaration that the resolver, the backend, the example harness, and the generated documentation all read. Omitted inferable dimensions replace explicit unknown markers; `name=` alone pins module identity; printing a module shows its resolved shapes. JSON is an internal graph/persistence format, not a third authoring frontend. PyTorch is an ordinary hard dependency of the package. Open interface decisions are listed at the end; implementation must settle those before dependent features or serialized formats ship.

## 1. Purpose and boundary

HNDL describes a neural network separately from its training code. Given input/output tensor contracts, a partially specified architecture, and explicitly selected construction policies, it resolves concrete dimensions or explains why resolution failed. It then builds ordinary PyTorch modules from the resolved plan.

```text
Author specification + tensor contracts + declared operators/policies
                              |
                           resolve
                              |
                Concrete plan + resolution trace
                              |
                            build
                              |
                     Ordinary torch module
```

The deliverable includes declarative Python configuration and trusted callable capture for sequences and static named graphs, bounded shape inference, a PyTorch backend, layer-by-layer inspection, and operator declaration through one decorator that built-in and custom operations share. A JSON-compatible graph representation underlies resolution and persistence. Object-based layer helpers are implementation details, not the primary public interface.

HNDL owns tensor contracts, architecture resolution, construction, operator registration, plan persistence, and diagnostics. The host owns data semantics, losses, metrics, regularizers, optimizers, training schedules, runtime random streams, devices, and complete checkpoint recovery. Researchers can vary architecture specifications while keeping their metric evaluation loop unchanged; HNDL does not optimize a metric or search architectures automatically.

Topology search, dynamic image sizes, runtime control flow, weight tying between graph nodes, distributed training, export backends, a visual editor, and full StyleGAN recipes are outside v1. HNDL has no HyperGAN runtime dependency.

## 2. Core invariants

1. **Literal values are constraints.** Resolution must never replace a specified width, stride, kernel, activation, or binding to make a network fit.
2. **Inference is not selection.** Uniquely determined values may be inferred. Choosing among valid architectures requires a named policy or more explicit information.
3. **Resolution allocates nothing.** Interpreting declarative configuration, applying declared relations, validating, resolving, and serializing must construct no module, allocate no tensor, use no CUDA, draw no randomness, and reach no network. Importing torch is not itself a violation: the invariant is about effects, not imports. The `pretrained` operator (§9) is the single documented exception and must remain the only one. Trusted callable capture executes user Python before this boundary and has no such side-effect guarantee.
4. **Successful plans are concrete.** All construction dimensions and operator arguments are resolved. Only the declared batch symbol may remain variable.
5. **Every resolved value has provenance.** Each argument records whether it was explicit, an operator default, policy-selected, or inferred, and the plan keeps the selected policy identity.
6. **Forward execution uses the plan.** It does not parse strings, solve constraints, select policies, or create parameters. Built-in operations draw no randomness during forward execution, with one documented exception: `dropout` in training mode (§6).
7. **Limitations are visible.** Ambiguity, unsupported inference, contradictions, and declared deviations must remain distinguishable.

## 3. Tensor contracts, ranks, and dtypes

A tensor contract is an ordered tuple of dimensions including batch, interpreted together with the plan's dtypes. Rank selects the layout; there is no separate layout field to keep consistent:

| Rank | Dimensions | Meaning |
| --- | --- | --- |
| 2 | `[B, F]` | Batch and feature width |
| 3 | `[B, T, D]` | Batch, `T` sequence positions, `D` features |
| 4 | `[B, C, H, W]` | Batch, channels, height, width |

`linear`, normalizations, and activations act on the last axis of rank-2 and rank-3 tensors, so a `[B, T, D]` sequence is mapped position by position. An operator that reads rank 3 as a one-dimensional signal — a 1-D convolution or pooling — must document that axis order as `[B, C, L]` in its own declaration and say so; the rank alone does not name the axes. Unsupported ranks fail rather than guessing. A shared batch symbol, written `"B"` in these examples, enforces equality across participating inputs and outputs. Concrete dimensions and runtime batch sizes must be positive integers. Only batch may remain symbolic after resolution; symbolic batch is preserved through reshape and flatten.

A plan carries a compute `dtype` of `float32` (the default), `float16`, or `bfloat16`, and an `input_dtype` that defaults to the compute dtype. The graph input may instead be an integer contract — `int64`, `int32`, or `bool` — for token ids and masks; a floating-point `input_dtype` must equal the compute dtype. Operator ports may declare their own dtype, including `any` for a port that accepts whatever its producer carries. Every edge's dtype is checked once during resolution and a mismatch fails with `E_DTYPE`, so an integer tensor cannot reach a floating-point port. The backend constructs parameters in the compute dtype and checks each port's dtype, shape, and device at runtime.

Both frontend APIs accept `input_shape` and `output_shape`, both including batch, plus optional `dtype` and `input_dtype` keywords. Structured graph contracts use the same tuples.

A contract is either one tuple or an ordered mapping of named contracts. `input_shape=("B", 128)` declares the single external input `x`; `input_shape={"z": ("B", 128), "y": ("B", 10)}` declares the external inputs `z` and `y` in that order. `output_shape` behaves the same way: a tuple declares the single public output `output`, and `output_shape={"logits": ("B", 1), "features": ("B", 256)}` declares two. `input_dtype` is either one dtype for every input or a mapping naming some of them, each defaulting to the compute dtype and following the integer/floating rules above per input. Port names match `[a-z][a-z0-9_]*`, are at most 32 per side, and may not collide with a registered operator alias (`E_NAME`). Every declared contract shares one batch symbol, inputs and outputs alike; a disagreement fails with `E_CONSTRAINT`. Every declared input must be consumed by the resolved graph or be selected as an output; an input no node reads fails with `E_BINDING` rather than being silently ignored. A mapping naming exactly `x` or exactly `output` is identical to the corresponding tuple contract, including in the saved plan.

Omitting an inferable dimension creates a fresh unknown. For example, `linear()` leaves output width to resolution, while `linear(64)` fixes it at 64. Both consume the current tensor unless one is supplied explicitly, as in `linear(x, 64)`. The operator declaration identifies inferable arguments; omission does not make every argument inferable. Structured JSON likewise omits such fields rather than storing a special sentinel. No public unknown-value helper is required.

| Author value | Meaning |
| --- | --- |
| Literal | A hard constraint; a mismatch fails |
| Omitted inferable dimension | Resolve through declared relations or an eligible selected policy; otherwise fail |
| Omitted optional non-dimension field | Use its versioned operator default, unless a selected policy requires a value |
| Omitted required field | Fail, unless a selected policy explicitly supplies that field |

An unresolved dimension must not quietly become an operator default when inference stalls. Derived input widths and channel counts are declared relations, not additional required user arguments; an operator may expose them as separate inferable arguments, such as `linear`'s `in_features`. `null`, `None`, `-1`, zero, arbitrary expressions, and keyword placeholders are not substitutes for omission. Each missing dimension is independent until a declared relation connects it.

HNDL does not infer color space, value range, activation choice, layout conversion, mixed precision, or broadcasting. A declared `tanh` is an explicit operation; an image output contract does not imply it. Joins must not insert resizing, casting, or broadcasting.

Activations are separate explicit graph operations in both frontends. Built-in `linear`, `conv`, and `deconv` contain no activation and accept no activation shorthand. An activation consumes its explicitly supplied tensor or current, following the same binding rules as other single-input operations. Thus `linear(64); relu(); linear()` applies ReLU only between the two linear layers. The resolver and construction policies must not insert, remove, or relocate activations, including at the network output. A composite operator — a transformer block, a residual block — may contain activations and normalizations internally as part of its declared semantics; that is one node, not resolver-inserted behavior.

## 4. Author specification and graph

The canonical internal model is a finite directed acyclic graph. Both frontends produce this graph before resolution. It represents named external inputs and multiple named public outputs, which both Python frontends declare through their contract arguments (§3); public authoring remains focused on those two frontends.

| Field | Contract |
| --- | --- |
| `schema_version` | Author data-format version; initially `1` |
| `inputs` | Ordered external inputs, each a name, a contract including batch, and a dtype. `input_shape`/`input_dtype` restate the first one |
| `dtype` | The compute dtype every floating port carries |
| `nodes` | Ordered node declarations with stable `id`, versioned `op`, `args`, input-port `inputs` bindings, and construction metadata |
| `outputs` | Ordered public outputs, each a name, the selected reference, and its required contract. `output_shape`/`output_ref` restate the first one |
| `frontend` | `python_config@1` or `python_callable@1`, with source provenance |
| `source` | Per-node annotations — location, selected policy identity, argument origins — excluded from numerical identity |

Input, output, and node IDs match `[a-z][a-z0-9_]*`. References are `input:<name>` or `node:<id>/<port>`. Unary operators consume `x` and produce `out` by convention; every operator declares its ports explicitly, and a variadic operator expands one declared prefix into `x0`, `x1`, and so on.

The following conceptual internal JSON encoding describes a two-layer perceptron whose last width is resolved from the output contract. Users author either Python frontend in §7. This illustrates the graph model; it is not accepted as source by `resolve()` and is not the serialized-plan schema in §10.

```json
{
  "schema_version": 1,
  "dtype": "float32",
  "input_dtype": "float32",
  "inputs": {"x": {"shape": ["B", 128]}},
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
    "output": {"ref": "node:head/out", "spec": {"shape": ["B", 10]}}
  }
}
```

Validation must reject duplicate IDs, missing references, undeclared ports, missing required bindings, cycles, nodes that cannot reach any selected output, and declared inputs that nothing reads. Unused output ports of an otherwise reachable node are allowed. Two public outputs may select the same reference, and a public output may select an external input directly. Graph values are tensors; an operator returning a Python tuple or dictionary must map it onto its declared tensor ports.

Each node owns an independent module instance and executes once per forward. Fan-out reuses a computed tensor. It does not clone the source module, repeat its invocation, or tie parameters between nodes. Explicit weight tying between graph nodes is deferred and must fail if requested. External reuse of built modules through sequence slices is supported as described in §7.

Both frontends lower to the declared external inputs and public outputs. With a single unnamed output, declarative configuration selects its final `out` binding when present and otherwise selects current, and a native callable may explicitly return a symbol or select current by returning `None`/falling through. With named outputs there is no default selection: configuration binds each declared name as a local and a callable returns a mapping of every declared name to one symbol (§7). Current initially refers to the first declared external input; its transitions are defined in §7. Every operator call creates one node in capture order. Explicit `name=` selects its stable node ID. Otherwise the ID is `n0`, `n1`, and so on, using the zero-based ordinal of that call among all operator calls, including explicitly named calls. Nested calls create their nodes in Python argument evaluation order. Duplicate explicit/generated IDs fail. Comments, local variable names, and assignment statements without new calls do not affect IDs. Inserting an operator call can change subsequent generated IDs; stable names are recommended for persisted models. Local variables refer to tensors and never infer node names.

## 5. Resolution algorithm

The resolver must perform the following logical steps:

1. Validate the schema and graph; bind exact operator identities from the active registry; order nodes topologically, breaking ties by declaration order; reject cycles and unreachable nodes.
2. Normalize each node's author arguments: apply the requirements of an explicitly selected policy, then operator defaults for still-omitted optional fields, then the declared per-argument validation. An explicit value contradicting a policy requirement fails with `E_POLICY_CONFLICT`. Record each argument's origin.
3. Check every edge's dtype once against the declared port dtypes (`E_DTYPE`), and seed the external input and public output contracts.
4. Sweep every node forward and then backward to a fixed point. Each sweep applies the declared shape patterns — shared symbols, integer scales, literals, and the shared ellipsis run — and then the operator's optional relation function, recording derivations and detecting contradictions.
5. Fail if any port shape is still incomplete (`E_AMBIGUOUS`) or any inferable argument was never determined (`E_UNRESOLVED`); otherwise run each operator's `finalize` hook, attach provenance, and freeze the concrete plan.

Rules may only add or refine facts. They must not retract constraints or overwrite literals; a relation that would rewrite a known value must fail instead. The resolver must bound graph size, expansion, dimension products, candidate sets, and sweep count. Exceeding a configured limit is an explicit failure, never silent truncation. Limits are the `limits` mapping accepted by every public entry point, with keys `max_nodes`, `max_edges`, `max_dimension`, `max_elements`, `max_state_bytes`, and `max_iterations`.

The required relation vocabulary includes positive integers, equality between ports, shared node-local dimension symbols, exact integer multiples, dimension literals, integer sums and differences for split and concat, element-count products for reshape and flatten, divisibility, bounded intervals, and fixed convolution equations. Direct equations, finite candidates, and bounded interval reasoning are sufficient; a general symbolic or SMT solver is not required.

Three outcomes must stay distinct:

| Outcome | Code | Meaning |
| --- | --- | --- |
| Contradiction | `E_CONSTRAINT`, `E_RESHAPE` | The supported relations prove the declared constraints incompatible |
| Ambiguous | `E_AMBIGUOUS` | Multiple valid values remain and no selected policy chooses among them |
| Unresolved | `E_UNRESOLVED` | Available rules cannot determine a required inferable argument |

An unsupported inverse is not proof that no architecture exists. A non-unique integer interval must not be resolved by picking its smallest member unless a selected policy explicitly requires that choice.

### Policies

A policy is declared by the operator that accepts it, as `policies={alias: Policy(identity, requires)}` in that operator's declaration. It has an immutable identity/version and a non-empty mapping of hard **requirements** over that operator's own arguments. Requirements enter before inference: they fill omitted eligible fields and reject contrary literals. They are not defaults, and no policy may relax a literal.

Precedence for an eligible field is: an explicit value and a policy requirement must agree, or the node fails; a requirement otherwise supplies the field; a still-omitted optional field takes its operator default; and only an argument that is still missing and declared inferable reaches inference. A policy therefore never competes with an inferred value, and policy order must not determine the result. The selected identity and every field it supplied are recorded in the plan, and a relation can read the selection through its node view so shape rules may depend on it.

The required initial policy is `spatial.up2_transpose@1`, declared by `deconv` (`conv_transpose2d@1`) and selected as `policy="up2"`:

| Property | Requirement |
| --- | --- |
| Applies to | Explicitly selected `conv_transpose2d@1` nodes |
| `kernel_size` | `4` on both spatial axes |
| `stride` | `2` on both axes |
| `padding` | `1` on both axes |
| `dilation` | `1` on both axes |
| `output_padding` | `0` on both axes |
| `groups` | `1` |

These settings guarantee exact spatial doubling; channel widths and bias are separate arguments. The policy does not guarantee image quality. It must not silently substitute resize/convolution, crop, or a different architecture when a target fails.

Custom operators declare policies the same way. Policy declarations are trusted Python that ships with the operator; data files cannot contain executable expressions or arbitrary imports, and configuration can only name an alias the selected operator already declares.

## 6. Built-in operators

The catalog is generated from the declarations themselves. [docs/operators](docs/operators/index.md), produced by `python -m hndl.docs` and checked in CI with `python -m hndl.docs --check`, is the published argument schema: for every operator it lists the alias, exact identity and version, ports and their shape relation, each argument's type, default, bounds and help text, and runnable examples with their resolved shape tables. This specification does not duplicate that table; it fixes the rules every built-in must obey.

- Every built-in is one `nn.Module` subclass with one `@operator(...)` declaration (§8), discovered from `hndl.operators` and bound by `Registry.builtins()`. Aliases and identities are unique; duplicate registration fails with `E_REGISTRY`.
- An alias binds one exact `identity@version`. Resolution never selects a newest installed version, and plans record the identity, not the alias.
- Unary operators use ports `x`/`out` unless their declaration says otherwise. Multi-input operators declare every port; multi-output operators declare port order.
- Batch equality propagates through every operator. Floating ports carry the plan's compute dtype unless the declaration says otherwise; no operator casts silently.
- Shape equations describe relations, not a license to accept arguments the backend rejects. Dimensions, channels, kernel sizes, strides, and dilations must be positive; padding and output padding must be nonnegative; group divisibility and backend argument restrictions must also hold.
- Activations are out-of-place and never mutate a shared branch. No built-in inserts resizing, broadcasting, casts, or layout conversion.
- `linear`, normalizations, and activations act on the last axis of rank-2 and rank-3 tensors. `linear` rejects rank-4 input with `E_CONSTRAINT` rather than flattening implicitly.
- `reshape` takes a prefix of leading non-batch dimensions and preserves the per-example element count. Three leading dimensions fix rank 4; a shorter prefix takes its rank from the neighbouring operations. Exactly one unknown factor can be solved from the element count; multiple unknown factors require independent relations or a selected policy. The plan records the complete resolved shape. `flatten` combines all non-batch dimensions into `[B, F]`.
- Convolution operates on `[B, C, H, W]` and uses the equations below independently per spatial axis.
- Join operators require exact agreement on every axis they do not combine. For `concat`, axes index the complete tensor shape; axis `0` is batch and cannot be concatenated, and negative axes are rejected rather than normalized.
- Operator defaults are versioned HNDL choices, not dynamically inherited backend defaults: `linear.bias=true`; convolution `bias=true`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`; transposed convolution additionally `output_padding=0`. Convolution `kernel_size` is required unless a selected policy supplies it. Output channels and linear widths are inferable when omitted and have no fallback default.
- Composite operators — attention, transformer blocks, residual blocks, mixtures of experts, feed-forward stacks — are ordinary declared operators that contain their activations and normalizations internally. They are one node with one module, resolve through their declared relation like any other operator, and appear in the generated catalog rather than here.

For each spatial axis, with input extent `I`, output extent `O`, kernel `k`, stride `s`, padding `p`, and dilation `d`:

```text
conv2d:           O = floor((I + 2p - d(k - 1) - 1) / s + 1)
conv_transpose2d: O = (I - 1)s - 2p + d(k - 1) + output_padding + 1
```

Ordinary convolution inversion may produce an integer interval, not a unique input; that interval is reported, intersected with other facts, and only fixed when it collapses. Transposed convolution does not inherently mean doubling. Each omitted dimension is a distinct unknown unless a declared relation connects it; known target height and width may separately determine two omitted seed axes.

### Randomness in `dropout`

`dropout` is the one deliberate exception to invariant 6. In training mode its `forward` draws an elementwise Bernoulli mask from torch's global RNG and rescales the survivors by `1 / (1 - p)`; in eval mode it is the identity. Two forward passes on the same input therefore differ unless the host reseeds between them, and a plan containing `dropout(p > 0)` is reproducible only under the host's own `torch.manual_seed` control. Everything the resolver sees stays deterministic: the mask never affects shapes, arguments, provenance, or either digest. The deviation must stay documented in the operator's own declaration, and no other built-in may draw randomness in `forward` without the same explicit treatment.

### Split and remainder

`split` takes input port `x`, output ports `first` then `rest`, an inferable `size`, and `dim` defaulting to `1`. It creates exactly two sections: the first `size` elements on `dim`, then all remaining elements. It does not repeatedly chunk the tensor at that size.

For input extent `I` on the selected axis and first section extent `s`, the relations are `first[dim] = s`, `rest[dim] = I - s`, and equivalently `I = s + rest[dim]`. Require integers `1 <= dim < rank` and `0 < s < I`; batch splitting, negative axes, and empty sections fail. Every other dimension, dtype, and batch value is equal across input and outputs. An omitted size is inferred from connected constraints when uniquely determined; otherwise ambiguity remains. No default chooses a section size.

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
linear(64)
relu()
linear()
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

The external symbolic input is prebound as `x` and is initially current. Every declared external input is prebound under its own name, so `input_shape={"z": ("B", 128), "y": ("B", 10)}` prebinds the locals `z` and `y`, and the first one is initially current. Calls with one input port may omit its tensor, and each single-output call makes its result current. This example needs no local assignments. An optional final `out` binding overrides the default current output; assignment never infers a module name. With named outputs `out` has no special meaning: each declared output name must be bound as a local holding one tensor symbol, as in `logits = linear(1)` and `features = h`, and a name that is never bound fails with `E_OUTPUT`.

`print(model)` uses the module's `__repr__`, following the standard PyTorch inspection convention, to display network input/output shapes and dtype followed by every layer's index, name, operation alias, and resolved input/output shapes in aligned columns. `repr(model)` returns the same representation. Inspection performs no forward pass, tensor allocation, or random draws:

```text
Network: [B, 128] -> [B, 10]  dtype=float32
index  name  operation  input shape  output shape
0      n0    linear     [B, 128]     [B, 64]
1      n1    relu       [B, 64]      [B, 64]
2      n2    linear     [B, 64]      [B, 10]
```

The index column is present only for a consecutive unary chain. A branched graph prints one row per node with every named port and its producing reference. The header appends `input_dtype=` whenever the graph input dtype differs from the compute dtype.

### Trusted Python callable

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

A callable with named inputs receives one keyword argument per declared input, so a two-input generator is written `def generator(*, z, y)`, and with named outputs it returns a mapping of every declared output name to one symbol. A plain tuple contract keeps the single positional parameter and the single returned symbol.

`from hndl import ops` exposes built-in symbolic operators with fixed version bindings. `registry.ops.<alias>` exposes operators bound in an explicit registry, including custom operations. These calls capture graph nodes, not numerical torch operations. Native Python variables follow ordinary Python rules and never name modules. The callable is invoked exactly once per API call, with one symbolic tensor for every declared external input, even if its body never refers to those parameters. With one unnamed output a returned symbolic tensor selects it; `None` or ordinary fallthrough selects current; any other return value fails without falling back to current. With named outputs the callable must return a mapping whose keys are exactly the declared names and whose values are symbols of this capture; `None`, a bare symbol, a missing name, and an undeclared name all fail with `E_OUTPUT`. Zero-argument callables are not a separate supported convention. A symbolic tensor cannot be used as a Python truth value or substituted with an eager tensor. Ordinary Python helpers and loops may construct a finite static graph; they do not become runtime graph control flow.

Symbolic factories allocate nothing. Capture context is scoped to that call, isolated across concurrent captures, and cleaned up after success or exceptions; it is not a mutable global registry. Symbols from another capture fail. Coercion to an integer, boolean, iterable tensor values, or raw eager torch execution is unsupported and must fail rather than sample concrete values. Iterating a declared tuple of output symbols is ordinary tuple handling, not iteration over a symbolic tensor. The function is never invoked during forward execution, printing, serialization, or restoration; function source inspection is unnecessary.

Callable capture is explicitly trusted Python execution. The function and registered declarations may have side effects, allocate tensors, import packages, or use randomness; HNDL does not sandbox them or claim that capture is pure. Hosts must choose trusted code accordingly. After capture, the shared resolver honors invariant 3. Capture errors propagate without retrying, executing source text, or silently switching frontend. Limits on the resulting graph do not bound arbitrary Python execution; hosts needing isolation must provide it outside this trusted API.

### Signatures and shared behavior

Every entry point below also accepts `limits=None`, the bounded resource mapping of §5.

| API | Result and requirements |
| --- | --- |
| `resolve(source: str, *, input_shape, output_shape, dtype="float32", input_dtype=None, registry=None)` | Interpret declarative configuration and return an immutable `ResolvedPlan` |
| `resolve_file(path, *, input_shape, output_shape, dtype="float32", input_dtype=None, registry=None)` | Bounded UTF-8 file loading followed by the same declarative resolution pipeline |
| `resolve_callable(fn, *, input_shape, output_shape, dtype="float32", input_dtype=None, registry=None)` | Invoke trusted capture once, then resolve the captured graph |
| `network(source: str, *, input_shape, output_shape, device, dtype="float32", input_dtype=None, registry=None, initialization_seed=None)` | Declarative resolution plus construction; `forward(x)` returns a tensor, and `forward(**inputs)` returns a tensor or a dictionary of named outputs |
| `network_file(path, ...)` | Bounded UTF-8 loading followed by the same declarative construction pipeline |
| `network_from_callable(fn, ...)` | Trusted capture, resolution, and construction with the same runtime interface |
| `model.plan` | Immutable resolved plan used to construct the module |
| `print(model)`, `repr(model)` | Complete layer/port shape table without sample tensors or execution |
| `print(plan)`, `repr(plan)` | The equivalent table without a backend: operator aliases and every named port's resolved shape in aligned columns |
| `plan.describe()` | The shape table plus the semantic digest and, per node, argument provenance and the canonical initialization/trainability records |
| `plan.to_json()`, `ResolvedPlan.from_json(text, *, registry=None)` | Canonical persistence and validated restoration (§10) |
| `hndl.torch.build(plan, *, device, initialization_seed=None, registry=None)` | Lower-level construction returning a dictionary-output `GraphModule` |
| `hndl.torch.parameter_counts(plan, *, registry=None)` | Per-node parameter counts from an allocation-free meta-device construction; `None` for a node that cannot be probed |

`resolve`, `resolve_file`, `resolve_callable`, `ops`, `Registry`, `ResolvedPlan`, `HNDLError`, and the declaration types `operator`, `Arg`, `Example`, and `Policy` are exported by `hndl`; construction functions are exported by `hndl.torch`. A string always means source, never a filename or function to execute. File paths are supplied only by the host through the explicit file APIs. A missing registry selects an independent built-in registry. Explicit registry objects supply extensions without a mutable global singleton. Built-in `ops` and explicit `registry.ops` calls capture exact operator identities; a capture using an identity unavailable in the selected resolution registry fails. Both modes use identical operator/policy rules and output contracts. Their graph has the external inputs and public outputs the contracts declare, defaulting to input `x` and output `output`; each output's contract constrains the tensor it selects. When a configuration with one unnamed output binds `out`, its final value must be one tensor symbol and takes precedence over current; otherwise current is selected. Native `None`/fallthrough selects current, while an explicit symbol selects itself and any other value fails. An unset current fails when default selection requires it; an invalid explicit selection always fails. Empty source or a callable with no operator calls using default selection chooses the external input, subject to matching input/output contracts.

Both network constructors accept one runtime tensor per declared external input — positionally in declaration order, by keyword, or a mix — and return one tensor for a single public output, or a dictionary keyed by the declared output names in declaration order for several. A single-input, single-output network is therefore still `model(x)` returning a tensor, even for a branched graph. Printed graphs include all input/output ports, including both split results; a shared producer appears once. The lower-level `build(plan)` builder always returns the dictionary keyed by public output names.

### Current input and output selection

Each capture starts with current pointing to its first declared external input symbol. Current is one reference, not an operand stack: `add()` never retrieves or pops earlier results and fails for missing explicit inputs. The current reference is capture-local authoring state, not a runtime tensor stored on the module. It is isolated across concurrent captures and restored correctly around nested captures and exceptions along with the registry context.

A one-input operator accepts an explicit leading symbolic tensor or its declared port keyword; otherwise it uses current. The declaration distinguishes that symbolic port from ordinary author values. For example, `linear(64)` uses current with width 64, `linear(z1, 64)` uses `z1`, and `linear(64, x=z1)` is the equivalent keyword binding. `linear(z1, x=z2)` duplicates the tensor binding and fails. Multi-input operators require every tensor port explicitly, even if only one is missing; there is no partial current default. A variadic operator requires contiguous bindings starting at `x0` and at least two tensors.

After a successful single-output operator call, its output becomes current. This applies to explicit-input calls, nested calls, and multi-input joins as well as implicit-input calls. A multi-output call such as `split(64)` returns its declared tuple and clears current; the next one-input operation must select an input explicitly, for example `linear(z1, 128)`. Merely unpacking or assigning `z1` does not restore current. A discarded multi-output result also leaves current unset; ordinary reachability rules still reject dead nodes.

Python argument expressions evaluate in ordinary order before an omitted input port is bound. For example, `add(linear(64), linear(64))` creates the first linear from prior current, the second from the first result, then joins their explicit results. It does not construct parallel linears from the same input. To branch from an earlier tensor, pass that saved symbol explicitly to each operator. No reevaluation or assignment introspection changes this ordering.

Assignments only store references; they never change current by themselves. `saved = x` preserves a reference, and `x = saved` rebinds a local without changing current. `out = saved` selects the configuration output without changing current for later calls. If `out` is present but invalid, fail instead of using current. In native entry callables, explicit symbol returns override current; only `None`/fallthrough uses current, and tuple/other returns fail. These output conventions apply to the entry point, not arbitrary Python helper functions. Passing a symbol to a native helper does not select current: its first operator must explicitly use that parameter if it is meant to start a different branch.

If default output selection finds no current, report `E_CURRENT`. Empty configuration or a callable with no operator calls using default selection chooses the unchanged input; identity is valid only when the supplied input/output contracts agree. No nodes or state entries are needed for that identity plan. Selected outputs do not excuse dead nodes: every constructed node must still reach a selected tensor.

### Named external inputs and outputs

A mapping contract (§3) declares several external inputs, several public outputs, or both. A conditional GAN discriminator reads an image and a label and publishes a score and the features behind it:

```python
from hndl.torch import network

source = """
plane = linear(y, 784, name="label_projection")
label = reshape(plane, 1, 28, 28, name="label_plane")
concat(x, label, name="conditioned")
conv(64, policy="down2", name="stage1")
leaky_relu(0.2)
features = flatten(name="features")
logits = linear(1, name="logits")
"""
model = network(
    source,
    input_shape={"x": ("B", 1, 28, 28), "y": ("B", 10)},
    output_shape={"logits": ("B", 1), "features": ("B", 12544)},
    device="cpu",
)
scores = model(image, label)          # {"logits": ..., "features": ...}
```

The equivalent callable takes the inputs as keyword arguments and returns the outputs as a mapping:

```python
def discriminator(*, x, y):
    label = ops.reshape(ops.linear(y, 784), 1, 28, 28)
    ops.concat(x, label)
    ops.conv(64, policy="down2")
    ops.leaky_relu(0.2)
    features = ops.flatten()
    return {"logits": ops.linear(1), "features": features}
```

The rules are the same in both frontends:

- Each declared input is one prebound tensor symbol named by the contract, and current starts at the first of them. A declared input that no reachable node reads and that no output selects fails with `E_BINDING`.
- Each declared output selects one symbol of that capture. Configuration binds the output's name as a local; a callable returns the mapping. Two outputs may select the same symbol, and an output may select an external input.
- Named outputs replace default selection entirely. `out`, a bare returned symbol, `None`, and fallthrough do not select a named output, and every declared name must be selected (`E_OUTPUT`).
- Printed shape tables and the header show every external port by name. A graph with one input and one output prints exactly as before.
- The header of a multi-port network reads `Network: x=[B, 1, 28, 28], y=[B, 10] -> logits=[B, 1], features=[B, 12544]  dtype=float32`, and `input_dtype=` names only the inputs whose dtype differs from the compute dtype.

### Declarative grammar and trust boundary

The declarative frontend has version `python_config@1`, recorded with source provenance. It interprets a pinned, allowlisted Python AST subset. The accepted grammar is the following contract, not whatever a newer Python parser happens to accept:

- The module contains simple assignment statements with one target and expression statements consisting of a registered operator call. Targets are a local name or a flat tuple/list of distinct names for unpacking. Chained, annotated, augmented, attribute, subscript, and starred assignments fail.
- Assignment values are existing symbolic tensor names, flat tuples of symbolic tensors, or registered operator calls returning those values. Nested registered calls are permitted in tensor arguments. Literal-only local variables are not part of v1; literals belong in call arguments.
- Calls name a registered operator alias directly, such as `linear(...)`. Attribute calls such as `ops.linear(...)` are reserved for trusted native Python and fail in configuration. Calls cannot be redirected through local aliases.
- Tensor ports take symbolic tensors, positionally in declared port order or through their declared keyword names. A single-input operator may omit that port and use current, with remaining positional literals bound to author arguments by its declaration. Multi-input operators require every tensor port explicitly; there is no partial default. Duplicate bindings and missing required ports fail.
- Argument literals are bounded integers, finite floats, booleans, strings, and schema-permitted `None`, plus recursively bounded literal lists, tuples, and dictionaries with unique string keys. Positive/negative numeric literal signs are allowed; arithmetic and computed expressions are not. Containers cannot conceal calls or arbitrary objects. Strings are accepted only where the declaration permits them; booleans are not valid integer dimensions. Inferable dimensions are omitted, not written as `None`.
- `name="hidden"` is reserved node metadata and `policy="up2"` selects a policy the called operator declares. `init` and `trainable` are reserved construction metadata (§11). None reaches an operator constructor as an ordinary argument, and a declaration may not name an argument or port `name`, `policy`, `init`, or `trainable`.
- There are no imports, function/class definitions, conditionals, loops, comprehensions, lambdas, arbitrary expressions, arbitrary function calls, attribute/subscript access, decorators, formatted strings, or star/keyword expansion. Expression statements other than registered operator calls, including bare names, literals, and docstrings, fail.

Python comments, whitespace, parentheses, and multiline calls follow the pinned grammar and preserve source locations for diagnostics. They do not change graph identity. Local names use ordinary non-keyword Python identifiers; special double-underscore names are rejected. Rebinding locals, including `x` and `out`, is allowed: evaluate the right-hand side against prior bindings, then bind the target. References to earlier tensors remain valid through other locals; rebinding does not mutate a tensor or rename/rebuild a node. Undefined and forward local references fail. Registered operator aliases cannot be rebound. The registry must reject aliases reserved for external `x` or output `out` in this frontend. If `out` is bound, its final value must hold one symbolic tensor, even if an earlier binding held a tuple. Ordinary local assignment or rebinding, including `x`, does not itself select current.

Tuple results may be stored in a local and later unpacked; unpacking must match the declared number of outputs exactly. For example, `parts = split(x, 64)` followed by `z1, z2 = parts` is valid. Tensor variables and module IDs are distinct: `h = linear(x, 64)` creates generated node `n0`, while `hidden = linear(x, 64, name="hidden")` creates node `hidden`. Renaming `h` cannot change the state keys. No assignment introspection is performed in either frontend.

### Loading limits and trust boundaries

String and file entry points share one pipeline. Apply byte/line bounds before parsing; files are read as bounded UTF-8 with explicit errors for invalid encoding or oversized content, without reading an unbounded file first. Apply common-indentation removal compatible with `textwrap.dedent` so indented triple-quoted examples work. Keep a source map from dedented columns to original line/column positions. Files may not request other files, imports, or callbacks. The host supplies a path; configuration cannot choose what to load.

The interpreter must never use Python `eval`, `exec`, or compiled-code execution on configuration. Restricting `__builtins__` around `exec` is not a security boundary, as the [Python execution documentation](https://docs.python.org/3/library/functions.html#exec) explains. Parse to an AST, validate the complete allowlist and budgets before interpreting it, then dispatch only through the explicit registry. Literal values are interpreted directly with bounded traversal, not passed to an unbounded evaluator. Pin the supported parser grammar/version with the frontend and reject unsupported syntax even if the host interpreter accepts it. Publish compatibility fixtures before release. No configuration-supplied callback, import path, class, or provider may be loaded.

Enforce bounds on AST node count/depth, literal/collection sizes, graph nodes/edges, and resolution work. Parsing itself can exhaust resources before AST validation, as documented for [Python AST parsing and literal evaluation](https://docs.python.org/3/library/ast.html). All declarative source, including file content, follows the same isolated parser/validator worker path with memory/time limits by default. Bound the worker's returned data and diagnostics too, and validate their format before accepting them. Do not infer that a string is trusted from its origin or silently fall back to an unbounded parser. A Python AST allowlist alone is not a denial-of-service sandbox. Operator declarations remain trusted code and must not be selected from arbitrary external imports by configuration. Use bounded, explicitly declared argument schemas, not user callbacks for interpreting values. Trusted native capture remains the separately requested execution mode, not an alternate parser behavior.

Registered parameter and buffer storage is bounded before real allocation. The builder first constructs every node on PyTorch's allocation-free `meta` device, sums the storage each node would own in the plan's compute dtype, and fails with `E_RESOURCE` if the running total exceeds `max_state_bytes`. A node whose meta construction is not possible is measured immediately after its real construction instead, so the bound still applies, and the build receipt records the resulting `state_bytes`. `parameter_counts(plan)` exposes the same allocation-free probe per node. A symbolic batch dimension prevents a complete runtime activation-memory estimate. Static bounds must not be presented as a guarantee of total training memory, and trusted operator implementations remain responsible for honoring their declarations.

### Operator calls and exact identities

Single-input calls accept an optional leading symbolic tensor, followed by author arguments: `linear(64)` and `linear(z1, 64)`, `conv(3, kernel_size=3)` and `conv(features, 3, kernel_size=3)`, `reshape(512)` and `reshape(features, 512)`, or `split(64, dim=1)` and `split(x, 64, dim=1)`. The declaration distinguishes symbolic port values from dimension/scalar literals; it does not guess that an integer is a tensor. The sole port may also be bound by its declared keyword. Required non-inferable arguments remain required. Multi-input calls such as `add(a, b)` and `adaptive_norm(features, params)` explicitly supply all tensor ports. Positional author arguments follow declaration order over the arguments marked positional; everything else is keyword-only.

A reshape prefix fixes the leading non-batch axes and bounds admissible output ranks. Length `K` requires at least `K` non-batch axes. `reshape(x, 512)` can describe rank 2, 3, or 4 until connected rules narrow it; `reshape(x, 512, 4, 4)` uniquely establishes rank 4. Structured `shape` arrays use the same prefix semantics. `reshape(x)` is valid when connected rules fix rank and dimensions; `[B,2,4,4] → reshape → linear` gives width 32. Ambiguous rank/factorization fails rather than selecting arbitrary dimensions.

Configuration calls and `ops` attributes contain no version suffixes. Registry aliases bind immutable exact identities: `linear` → `linear@1`, `conv` → `conv2d@1`, `deconv` → `conv_transpose2d@1`, `split` → `split@1`, `adaptive_norm` → `adaptive_norm@1`, and `deconv`'s policy `up2` → `spatial.up2_transpose@1`. Duplicate/conflicting registrations fail. Plans preserve exact identities and policy effects, so restore does not depend on current alias bindings.

### Branches and reusable subnetworks

This configuration takes `[B,128]` and returns `[B,128]`:

```python
source = """
# Split clears current; explicitly choose the next branch.
z1, z2 = split(64, name="partition")
linear(z1, 128, name="content")
features = relu(name="features")
style = linear(z2, 128, name="style")
add(features, style, name="combined")
"""
```

The built-in adaptive normalization can use the remainder directly:

```python
source = """
z1, z2 = split(64)
linear(z1)
features = reshape(32, 4, 4)
adaptive_norm(features, z2)
"""
```

Input `[B,128]` and output `[B,32,4,4]` infer projection width 512, with the 64-feature remainder matching `2 * 32` style parameters. Custom aliases must be registered before interpretation. In native code the equivalent call is `registry.ops.adaptive_norm(features, z2)`.

A reusable subnetwork is an ordinary declared operator such as `style_mlp`, with its own ports, shape relation, and module. Configuration calls `style = style_mlp(z2)`; native code calls `style = registry.ops.style_mlp(z2)`. Each invocation creates an independent node/module. Trusted native helper functions may themselves compose symbolic ops; configuration does not define or call arbitrary Python helpers.

### Layer access

A lowered consecutive unary chain provides indexing, negative indexing, name lookup, length, and iteration over the actual registered layers:

```python
first = model[0]
last = model[-1]
first_by_name = model["n0"]
assert first is first_by_name
assert len(model) == 3
layers = list(model)
features = model[:2]
```

Module lookup uses explicit or generated node IDs, never local variable names. Unknown IDs raise `KeyError`; out-of-range integer indexes raise `IndexError`. A slice returns ordinary `torch.nn.Sequential` over the existing module objects, sharing parameters, buffers, mode, and device moves. It carries no resolved-plan guarantees. Slicing requires adapters compatible with direct tensor-in/tensor-out chaining; otherwise reject it explicitly. Creating a slice does not alter registrations/state keys on the original model.

Registration remains once under `nodes.n_<node_id>`, without a facade prefix or duplicate registrations. Structural replacement/insertion/deletion through the HNDL container is rejected; ordinary parameter updates remain supported. Architecture changes require a new resolution/build. Actual split/fan-out/join graphs support node-name lookup only, regardless of which frontend authored them. Both network constructors still return the single selected tensor at runtime when the graph declares one public output; `build(plan)` returns a dictionary-output GraphModule. Indexing, slicing, iteration, and `len` require a consecutive unary chain, which means exactly one external input and one public output; a graph with named ports on either side supports node-name lookup only.

### Worked generator resolution

```python
from hndl.torch import network

source = """
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
model = network(
    source,
    input_shape=("B", 128),
    output_shape=("B", 3, 32, 32),
    device="cpu",
)
print(model)
```

Complete shape output:

```text
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

The last convolution preserves spatial size. Three doubling stages give `output_height = 8 * seed_height` and the same for width; projection width is `512 * seed_height * seed_width`.

| Required output | Resolved seed | `n0.out_features` | Outcome |
| --- | --- | --- | --- |
| `[B,3,32,32]` | `[B,512,4,4]` | `8192` | Resolved |
| `[B,3,64,64]` | `[B,512,8,8]` | `32768` | Resolved |
| `[B,3,32,64]` | `[B,512,4,8]` | `16384` | Resolved |
| `[B,3,30,30]` | Would require axes of `3.75` | — | Contradiction |
| `[B,3,32,32]` with literal projection width `128` | Needs `8192` elements | — | Contradiction; retain the literal |

## 8. Custom operators and minimal graphs

An operation is declared once, on the `nn.Module` subclass that implements it. The declaration is the single source of truth for the resolver (ports, shape relation, arguments), the backend (the class itself), the test harness (examples and the optional reference implementation), and the generated documentation (summary, docstring, help text). Built-in and custom operations use exactly the same decorator; there is no second registration step, no separate shape-rule object, and no backend-binding call.

| Field | Contract |
| --- | --- |
| `alias` | The name configurations call. A non-keyword identifier other than `x`/`out`, without a leading underscore. Aliases are reserved words inside a configuration |
| `identity`, `version` | Exact immutable identity, defaulting to the alias, and a positive semantic version defaulting to `1`. Lowercase dotted identifiers; a namespace such as `example.silu` is recommended for custom operators and no longer required |
| `summary` | One sentence for the catalog index. The class docstring is the long Markdown description |
| `shape` | The port/relation DSL below. Required |
| `relation` | Optional callable receiving the bounded node view, for rules the DSL cannot express. `shape_text` gives the docs a one-line description of it |
| `args` | Ordered mapping of scalar argument names to `Arg` declarations. Every `Arg` needs help text |
| `examples` | Runnable `Example(source, input_shape, output_shape, note="", input_dtype=None, network=False)` entries |
| `category` | Short lowercase grouping used by the generated index |
| Hooks | `positional_rest`, `policies`, `validate`, `finalize`, `reference` (below) |

Registration fails with `E_REGISTRY` for a duplicate alias or identity, a missing summary or help text, a malformed shape string, an argument colliding with a port or with reserved metadata (`name`, `policy`, `init`, `trainable`), or a constructor that cannot accept the declared arguments. Each declaration permits at most 32 input ports, 32 output ports, 64 scalar arguments, and 64 distinct dimension symbols; port, argument, and dimension names are at most 64 characters; dimension literals and scales are at most `2**31 - 1`; integer arguments satisfy `abs(value) <= 2**63 - 1`; string arguments are at most 16,384 UTF-8 bytes.

### The shape DSL

`shape="x[B, ..., D_in] -> out[B, ..., D_out]"` declares input ports before the arrow and output ports after it, in binding order. The grammar is finite data, not an expression language:

- `name[dims]` declares a port and its pattern; `name` alone declares a port the DSL does not constrain, to be paired with `relation=`. The first input port receives the implicit current tensor.
- `B` is the batch axis and must be first. Other entries are dimension symbols (`C`, `D_in`), exact integer multiples written `2*C`, or positive integer literals. A symbol is scoped to one node invocation: the same name on two ports or axes asserts equality in both directions, and unrelated nodes never share a symbol because both spell it `C`.
- `...` stands for the same run of middle axes on every port that uses it, so `x[B, ...] -> out[B, ...]` preserves any supported rank. It may appear once per pattern, after the batch axis. A pattern without `...` must have a supported rank (2, 3, or 4) including batch.
- `x*` declares a variadic input expanded to `x0`, `x1`, … . A variadic operator must be the sole input port and must declare an `int` argument named `input_count`.
- `port:dtype` declares a port dtype: `compute` (the default, the plan's compute dtype), `any`, `float32`, `float16`, `bfloat16`, `int64`, `int32`, or `bool`. Edge dtypes are checked once per edge at resolution (`E_DTYPE`).

Scaling is exact: an extent of 65 cannot satisfy `2*C`. Contradictions fail; dimensions that no relation determines remain ambiguous rather than receiving guessed values.

### Argument declarations

`Arg(type, default, *, min, max, exclusive_min, exclusive_max, help, inferable, dim, positional, choices)` declares one scalar argument. Supported types are `int`, `float`, `bool`, `str`, `"pair"` (an int or two ints, normalized to a pair), and `"ints"` (a bounded tuple of ints). Integers and booleans are distinct; float arguments accept finite integer or float literals and normalize to floats; `choices` restricts a `str`.

Omitting `default` makes the argument required unless `inferable=True`, which lets the resolver solve it and forbids a default. `dim="D_out"` ties an int argument to a shape symbol so a known extent fills the argument and an explicit argument fixes the extent; two arguments cannot bind to the same symbol. `positional=False` keeps an argument keyword-only; positional order is otherwise the mapping order, after any tensor ports. Numeric bounds are inclusive unless the matching exclusive flag is set, bounds must admit a value, and a declared default passes the same validation as an explicit one. Every `Arg` must carry non-empty `help`; the catalog publishes it.

### Hooks

- `positional_rest="shape"` routes leftover positional literals into one `"ints"` argument, which is how `reshape(512, 4, 4)` works.
- `policies={"up2": Policy("spatial.up2_transpose@1", {...})}` declares the policies this operator accepts; requirements may only name its own arguments (§5).
- `validate=fn(args)` performs cross-field checks on normalized arguments before resolution, raising `HNDLError` with an explicit code.
- `finalize=fn(args, input_shapes, output_shapes)` stores resolved values back into the plan arguments once shapes are concrete. It must be idempotent and must not introduce a value the solver did not prove.
- `reference=fn(module)` returns a handwritten equivalent the test harness compares against.

### The node view given to a relation

A relation is a Python function that runs on every solver sweep, forward and backward, and may only add facts. It sees one bounded facade, never the solver internals or other nodes:

| Member | Meaning |
| --- | --- |
| `s.args` | Mutable normalized arguments of this node |
| `s.inputs`, `s.outputs` | Declared port names in order; `s.spec` is the operator declaration |
| `s.policy` | The selected policy identity, or `None` |
| `s.shape(port)` | Known dimensions of a port, with `None` for unknown entries, or `None` |
| `s.dtype(port)` | The dtype name carried by that port's edge |
| `s.rank(port, r)` | Assert a rank and return the shape list |
| `s.axis(port, i, value, code="E_CONSTRAINT")` | Assert one extent |
| `s.equal(p, q)` | Assert two ports have the same shape |
| `s.arg(name, value)` | Fill or confirm an argument, failing on a conflicting explicit value |
| `s.product(p, q)` | Assert equal per-example element counts, solving one unknown factor |
| `s.interval(port, i, lo, hi)` | Intersect a bounded inverse interval, fixing the axis when it collapses |
| `s.error(code, message)` | Fail with a stable code and this node's source location |

Referencing an undeclared port fails with `E_REGISTRY`. Relations must terminate and must be monotone; the resolver bounds sweeps with `max_iterations` and fails with `E_RESOURCE` rather than looping. Shared relations for families of operators — the convolution arithmetic, elementwise joins — live alongside the operators that use them.

### Registration, trust, and persistence

```python
from torch import nn
from hndl import Registry
from hndl.torch import network

registry = Registry.builtins()


@registry.operator(
    "silu",
    identity="example.silu",
    version=1,
    summary="Sigmoid linear unit, x * sigmoid(x).",
    shape="x[B, ...] -> out[B, ...]",
)
class SiLU(nn.SiLU):
    """Out-of-place SiLU. No parameters, no buffers, identical in train and
    eval mode."""


model = network(
    """
linear(64)
silu()
linear()
""",
    input_shape=("B", 128),
    output_shape=("B", 10),
    registry=registry,
    device="cpu",
)
```

`Registry.builtins()` returns an independently extensible registry containing the built-in catalog. `@registry.operator(...)` declares and registers in one step; `registry.add(cls)` registers a class already decorated with `hndl.operator`. `registry.get(alias)` and `registry.by_identity("example.silu@1")` are the exact lookups resolution and restoration use. A shape pattern of `x[B, ...] -> out[B, ...]` contributes bidirectional equality for rank and every dimension, so downstream constraints propagate backward through the custom layer; it is not a forward-only shape callback.

Declarations are trusted code, for custom operators as much as for built-ins: relation functions and module constructors run in the host's process. A registered relation is allowed to be arbitrary Python, and custom operators may therefore express any rule a built-in can. What remains untrusted is data: configuration source and saved plans can name only an alias or identity the host already registered, can never import a class, a module path, or a callback, and are bounded by the same parser and resolver limits. A plan stores `example.silu@1` plus concrete arguments and port shapes, not executable Python, and building or restoring it without that registration fails with `E_STATE_VERSION`.

The constructor receives every declared argument by keyword, plus any shape symbol it names as a keyword-only parameter (`*, D`) and, when requested by name, `input_shapes` / `output_shapes` mappings of resolved port shapes. It runs under `torch.device(device)`, so tensors may be created normally; parameters are then cast to the plan's compute dtype. `forward` receives tensors positionally in declared input-port order and returns one tensor, or a tuple/list in declared order or a dictionary with exactly the declared names for multiple outputs. HNDL validates each result's shape, dtype, and device. Within a built model each node owns an independent module instance and registered state; reusing a module or registered storage across nodes is rejected. Sharing a producer tensor across branches, or obtaining a slice of an already built chain, remains supported. A genuine unary chain supports indexing and shared sequential slices even when its ports use names other than `x` and `out`.

Saved plans contain concrete shapes and arguments plus exact operator identities and versions. Declarations remain in the explicitly supplied registry; artifacts cannot import them. Restoring a custom plan revalidates its concrete equations against that registry, and any mismatch fails with `E_INTEGRITY` rather than being re-inferred. An operator must raise its semantic version when it changes a shape relation, an argument schema, its numerical meaning, or its parameter layout.

### Worked example: adaptive normalization

The built-in `adaptive_norm@1` demonstrates the multi-port case. Its declaration is the whole contract:

```python
@operator(
    "adaptive_norm",
    summary="Instance-normalize features, then apply a per-example learned scale and bias.",
    shape="x[B, C, H, W], params[B, 2*C] -> out[B, C, H, W]",
    args={"eps": Arg(float, 1e-5, min=0, exclusive_min=True,
                     help="Added to the variance for stability.")},
    category="normalization",
)
class AdaptiveNorm(nn.Module):
    ...
```

`2*C` resolves an omitted style-affine output width in either direction, and the shared `C` and the repeated `x`/`out` pattern give batch, channel, and spatial equality without a relation function. The numerical convention is:

```text
mean       = mean(x, spatial_axes, keepdim=True)
variance   = mean((x - mean)^2, spatial_axes, keepdim=True)
normalized = (x - mean) / sqrt(variance + eps)
[delta_gamma, beta] = split(params, C, axis=features)
out = (1 + delta_gamma)[B,C,1,1] * normalized + beta[B,C,1,1]
```

`eps` must be positive, defaults to `1e-5` for version 1, and is explicit in the plan. Variance is population variance. Both feature and style paths are differentiable. The operation has no running statistics and identical train/eval behavior.

A style-conditioned graph exercises fan-out through it: a mapping MLP produces `w`; `w` fans out into a feature projection/reshape and a style affine; both feed the normalization node. With external input `z: [B,128]` and output `features: [B,64,4,4]`:

| Node | Operator and author arguments | Input bindings | Resolved output |
| --- | --- | --- | --- |
| `mapping` | `linear@1`, `out_features=256` | `x=input:z` | `[B,256]` |
| `w` | `relu@1` | `x=node:mapping/out` | `[B,256]` |
| `project` | `linear@1`, output width omitted | `x=node:w/out` | `[B,1024]` |
| `seed` | `reshape@1`, `shape=[64,4,4]` | `x=node:project/out` | `[B,64,4,4]` |
| `style` | `linear@1`, output width omitted | `x=node:w/out` | `[B,128]` |
| `norm` | `adaptive_norm@1` | `x=node:seed/out`, `params=node:style/out` | `[B,64,4,4]` |

Both branches consume the same `w` tensor; `mapping` and `w` each execute once. Reshape determines the projection width as `64 * 4 * 4`; the `2*C` relation determines the style width as `2 * 64`. The executable example declares `init={"weight": 0, "bias": 0}` on its style affine, giving unit scale and zero bias on normalized features at initialization; arbitrary linear layers do not inherit that initializer, and rebuilding the saved plan reapplies it. Save and reload the state dictionary to preserve trained values.

This is an AdaIN-style demonstration, not a faithful StyleGAN implementation. Learned constants, noise blocks, and modulated convolutions are separate operators. Stochastic operators other than `dropout` should receive caller-owned noise through tensor ports.

## 9. Pretrained checkpoints

`pretrained(source, output=, component=, config=)` loads an existing checkpoint as one node of an HNDL network. The checkpoint's own configuration defines the architecture: HNDL does not describe GPT-2 or ResNet layouts. A `config.json` with a `model_type` is built by `transformers`, one with an `architecture` by `timm`, and HNDL wraps the result as a single node with an inferred output contract. Nothing in a checkpoint is executed as code, and the operator requires the optional `hndl[pretrained]` extra; a missing dependency, an unusable source, or a configuration that names neither library fails with `E_PRETRAINED`.

| `source` | Meaning |
| --- | --- |
| `hf://owner/repo` | A Hugging Face Hub repository at its default branch |
| `hf://owner/repo@revision` | A branch, tag, or commit of that repository |
| `/path/to/checkpoint` | A directory holding `config.json` and safetensors weights |
| `/path/to/model.safetensors` with `config="/path/to/config.json"` | A bare weights file plus its configuration |
| `/path/to/weights.pth` with `provider=` and `sha256=` | A local state dict loaded into a host-registered architecture |

A `.pth` state dict describes no architecture, so it names one that trusted host code registered: `registry.pretrained_provider(name, build)` binds a zero-argument callable returning the `nn.Module`, and configuration may only name a provider already present on the registry the resolution and build run under. Providers are per-registry, so one `Registry.builtins()` never inherits another's. This is the same trust boundary as operator declarations: configuration selects no import path, class, or callable. `sha256` is required for such a source and is verified against the file both at resolution and immediately before loading, which fails with `E_PRETRAINED` on a mismatch; the file is read with `torch.load(map_location="cpu", weights_only=True)` — no object is unpickled — and loaded with `strict=True`, so missing or unexpected keys fail with `E_PRETRAINED`. `layer` names a submodule by its dotted `named_modules()` path; the node runs the model with a forward hook there, returns that submodule's output, and stops the pass, and an unknown name fails with `E_PRETRAINED` listing the available submodules. Omitted, the node returns the model's own output. `readout` names a callable the host registered alongside the builder — `registry.pretrained_provider(name, build, readouts={...})` or `registry.pretrained_readout(provider, name, fn)` — which the node calls as `readout(model, x)` in place of the model's own forward; configuration may only name an already registered readout, an unknown name fails with `E_PRETRAINED` listing the ones the provider offers, a readout that returns anything but a single tensor fails with `E_PRETRAINED`, and `readout` and `layer` are mutually exclusive. A readout runs during resolution on the meta device, so it must be a pure function of its arguments. A provider source fixes no input contract beyond requiring a floating-point input; the meta-device trace validates the declared shape.

This operator is the documented exception to invariant 3. Its relation reads `config.json` — downloading only that file for an `hf://` source when it is not cached — or hashes the pinned `.pth` file, and traces the architecture on PyTorch's `meta` device with the declared input shape to learn the output shape. No weights are read, no tensor is allocated, and no randomness is drawn during resolution; weights are downloaded and loaded when the network is built, and the allocation-free `parameter_counts` probe does not open a local checkpoint at all. The exception is confined to this operator: no other operator may read a file, reach the network, or trace a module during resolution.

The input contract follows the checkpoint. A text model consumes token ids, so the graph input must be `("B", T)` with an integer `input_dtype`; a floating-point input fails with `E_DTYPE`. A vision model consumes `[B, C, H, W]` at the checkpoint's native resolution, which the relation fixes, so a different size fails with `E_CONSTRAINT`. `output` selects the returned tensor — `"features"` (the default), `"pooled"`, `"logits"`, `"embeds"`, or a raw output attribute name — and `component="vision"` or `"text"` picks one tower of a multi-modal checkpoint.

The resolved commit hash or content digest is recorded as the node's `revision`, an inferable argument with provenance `inferred`. It participates in the semantic digest, and restoring a saved plan re-resolves the source: if it now points at a different checkpoint the restore fails with `E_CONSTRAINT` rather than silently loading other weights.

The wrapped model is frozen (`requires_grad=False`) and stays in eval mode even while the surrounding network trains, so the checkpoint's dropout and batch statistics do not move; `model.train()` does not override that. Passing `trainable=True` unfreezes it and lets train mode reach it. Weights are loaded in float32 and cast to the plan's compute dtype. The node participates in the ordinary `max_state_bytes` bound, appears in the shape table like any other node, and its examples are marked as network-dependent so offline test runs skip them.

## 10. Resolved plans and reproducibility

A `ResolvedPlan` must be immutable and include:

- `schema_version` and `resolution_version`; exact operator identities and versions; `frontend` provenance for configuration or callable capture.
- `input_shape`, `output_shape`, `output_ref`, `dtype`, and `input_dtype`, restating the first external input and the first public output.
- The ordered external `inputs` (name, contract, dtype) and public `outputs` (name, reference, contract) whenever they are not exactly the single input `x` and the single output `output`. A plan with only those two omits both fields, so its encoding and digests are unchanged.
- Concrete nodes in stable execution order, each with `id`, `op`, resolved `args`, `inputs` bindings, `outputs`, `input_shapes`, `output_shapes`, and the canonical `initialization`/`trainability` records.
- Per-node `source` (locations, selected policy identity) and `provenance` (each argument's origin).
- Semantic and artifact digests.

A node record carries no state-size or state-version field: storage is bounded at build time by the meta-device pass (§7), and parameter layout compatibility is carried by the operator's semantic version.

Serialization uses canonical JSON with sorted object keys, compact separators, UTF-8 without ASCII escaping, arrays for tuples, finite numbers only, and SHA-256 digests. Plan JSON is bounded at 16 MiB. Reading rejects duplicate JSON fields, unexpected or missing plan fields, non-finite constants, and nodes missing any canonical field (`E_SCHEMA`), then verifies both digests (`E_INTEGRITY`) and revalidates the concrete equations against the supplied registry without re-running source, a callable, or a policy choice. Exact encoding and digest test vectors must be published before the format is frozen.

The **semantic digest** covers the concrete graph: schema and resolution versions, node identities and order, operator identities, arguments, port contracts, initialization, trainability, the graph contracts, and both dtypes. It excludes frontend provenance, source locations, and inference explanations. Policy effects are expanded before hashing, so equivalent concrete numerical plans have identical semantic identity regardless of authoring syntax or derivation.

The **artifact digest** covers the complete saved plan except its own digest field. It includes provenance and frontend information that can differ between numerically equivalent plans. Parameter counts, where statically available, are not estimates of peak training memory.

## 11. PyTorch backend contract

`build(plan, *, device, initialization_seed=None, registry=None, limits=None)` returns a normal training-mode `GraphModule`. The network constructors present the single-input/single-output facade from §7 over the same construction semantics and registered state, with sequence indexing only for unary chains. Device is explicit; CPU is suitable for small fixtures and CUDA is the intended primary execution target. Any other device type fails; there is no silent CPU fallback and no silent device index substitution. The plan fixes the compute dtype.

- Construction revalidates the plan, then runs an allocation-free `meta` pass over every node to bound registered storage before allocating anything (§7). The build receipt records `torch_version`, `device`, `dtype`, `initialization_seed`, `seed_mode`, and `state_bytes`.
- Parameters and buffers are constructed in the plan's compute dtype. A node is constructed under the target device so constructors may allocate normally.
- A `GraphModule` validates declared external input contracts in `forward(**inputs)`, which accepts exactly the declared input names, and returns a dictionary keyed by public output names in declaration order, including for a single output. A missing, repeated, or undeclared runtime input fails with `E_BINDING` before any node runs. Both network facades instead accept the declared inputs positionally in declaration order, by keyword, or both, and return the selected output tensor for a single public output or the same dictionary for several, preserving the contract checks even for a branched graph.
- Every port is checked for shape, dtype, and device before and after applying its node, and a module that creates or removes registered state during forward fails with `E_RUNTIME`.
- Modules register once under `nodes.n_<node_id>`; the prefix avoids collisions with module attribute names. Stateless nodes keep execution/diagnostic identities without state entries. Two nodes sharing a module instance, tensor, or storage fail with `E_REGISTRY`.
- Nodes run in stable topological order, breaking ties by declaration order. The order is saved in the plan.
- Parameters/buffers are fully materialized before optimizer or distributed setup. No first-forward parameter creation is allowed.
- Runtime must preserve ordinary gradients. It must not silently detach, clone, cast, move tensors, or mutate a shared branch through in-place activations.

Initialization rules belong to the plan; the build receipt records the selected seed mode and runtime. Initial support is versioned `torch_default@1` plus explicit constant overrides. Both network constructors and `build` default to `initialization_seed=None`: construction uses and advances the caller's ordinary PyTorch RNG state, so `torch.manual_seed(...)` controls initialization as it does for handwritten modules. Supplying an explicit integer instead uses an isolated RNG scope on the requested device and restores the caller's RNG state afterward. No seed argument is needed for ordinary use.

The supported PyTorch runtime must be recorded because its defaults affect initialization. Module construction order is stable; operator constructors must respect the chosen RNG mode. Exact initial tensors across devices or PyTorch versions are not promised. Initialization draws happen at construction only; this does not permit hidden random draws during representation or inspection, and `dropout` (§6) is the only declared forward-time exception.

An explicit initialization seed scopes module construction only. It neither isolates nor controls randomness or other effects inside an author callable, which runs earlier during trusted capture.

Per-node trainability masks are applied during construction and saved in the plan. Freezing parameters does not select evaluation mode; the host controls train/eval behavior and must preserve these masks. An operator that documents its own mode behavior — the frozen `pretrained` node staying in eval mode — must state that in its declaration. Stateful modules declare persistent buffers and manage derived caches correctly after state loading/device moves.

### Initialization and trainability declarations

Both frontends accept optional reserved metadata on any operation:

```python
linear(64, trainable=False)
relu()
linear(init={"weight": 0, "bias": 0}, trainable={"bias": False})
```

`init` is a mapping from exact relative parameter names to constant numbers. Omission or an empty mapping retains ordinary constructor initialization, identified in the plan as `torch_default@1`. Overrides apply after the module is materialized on the selected device, under `torch.no_grad`, within the construction RNG scope. Constructors still execute their normal initialization and consume their normal RNG draws. Overrides draw no randomness and do not affect forward-time execution.

`trainable` accepts a boolean applying to every recursive parameter or a mapping from exact relative parameter names to booleans. Omission or an empty mapping preserves the constructor's `requires_grad` flags; this also preserves deliberately frozen parameters, such as those of a `pretrained` node. Unspecified parameters in a mapping retain those flags. An explicit boolean applies even to parameters the constructor froze. A boolean on a parameterless operation is valid and has no effect. Explicit `None` for either author keyword is invalid.

Each mapping has at most 256 entries. Paths have at most 256 ASCII characters, with dot-separated nonempty identifier or numeric segments, supporting names such as `projection.weight` and `layers.0.weight`. They are lookup keys into registered parameters, never Python attribute expressions to evaluate. Wildcards, indexing expressions, and buffer targets are unsupported. Values for `init` must have exact type `int` or `float`, be finite, and round to finite IEEE binary32 values; booleans are rejected. The plan stores the rounded float32 value, including underflow to zero, preserving the sign of zero, and the backend fills it into the parameter's compute dtype. Trainability values have exact type `bool`.

Capture and resolution validate these bounded declarations but cannot confirm a module's parameter inventory. The backend verifies every target after construction and before applying any overrides; missing parameters (including disabled `bias` or `affine`), buffer targets, and conflicting aliases fail. Multiple names of the same parameter must not receive conflicting explicit values or flags. Constant initialization requires a materialized parameter in the plan's compute dtype and rejects storage shared with any distinct parameter or registered buffer. When trainability settings touch shared storage, all distinct parameters using that storage must have the same effective flag; untouched constructor defaults remain valid. A request to enable gradients on an integer parameter fails. Module/state ownership checks remain in force across nodes.

The immutable node records persist canonical construction fields separately from operator arguments:

- `initialization = {"kind": "torch_default@1", "overrides": {...}}`.
- `trainability = {"default": null | boolean, "overrides": {...}}`, where null preserves constructor flags. A boolean default and named overrides are not combined by the author API.

These fields participate in semantic and artifact digests and survive concrete-plan validation. Omitted options and empty mappings have identical semantic identity. Settings that affect constructor behavior are deliberately retained even when a particular module has no parameters. Changing values or `requires_grad` flags on a built module does not mutate its plan.

Build and restoration apply the persisted settings before returning the module. Freezing does not detach the layer's output: input gradients still propagate. The resulting module starts in training mode; calling `train()` or `eval()` does not alter trainability. A PyTorch state dictionary contains parameter values and persistent buffers, not these construction declarations or the trainability mask; rebuild the matching plan before loading state. Rebuilding resets values to the declared initializer until a state dictionary is loaded.

## 12. Persistence and compatibility

Persist the full resolved plan, plus declarative source or trusted callable provenance as applicable. Do not pickle a callable or closure as the architecture. Restoration must never execute the author callable or configuration to reconstruct saved state. Reconstruct from the saved plan, not by applying a newer resolver to old omitted dimension fields. Before state loading, validate the digests, required implementation versions, and state names/shapes/dtypes.

The implementation writes and reads resolved-plan schema **1**, with resolution semantics version **1**. Initialization and trainability are part of this format: every saved node requires both canonical construction fields, including defaults. Unsupported versions fail with `E_STATE_VERSION`; missing canonical fields fail with `E_SCHEMA`; inconsistent contents fail with `E_INTEGRITY`. The conceptual author graph in §4 is separate from this persistence format.

These identities serve different purposes:

| Identity/version | Purpose |
| --- | --- |
| Package version / implementation revision | Identify implementation provenance |
| Plan schema version | Decode saved structures |
| Resolution-semantics version | Identify planning behavior |
| Operator/policy semantic versions | Preserve mathematical and construction meaning, including parameter layout |
| Frontend version (`python_config@1`, `python_callable@1`) | Identify the authoring grammar and its provenance |
| Host checkpoint compatibility version | Decide complete training continuation compatibility |

An older semantic version must retain its meaning or fail with the affected version and an actionable compatibility/migration explanation. A package version alone is not proof of numerical compatibility. An architecture change requires a new model or explicit weight transfer; it is not ordinary checkpoint continuation.

Adapters for host applications may select one named output tensor and save the HNDL plan alongside training state. Optimizer, EMA, RNG, data position, distributed execution, and complete recovery remain host responsibilities. Construction success does not qualify a network for a host's numerical or distributed guarantees.

## 13. Diagnostics

Failures must expose a stable code, source/node/field location, affected constraint chain, and suggested explicit remedies. Suggestions must never modify the specification automatically. The catalog is closed: every failure raises one of these codes.

| Code | Example |
| --- | --- |
| `E_CONSTRAINT` | Target `30` conflicts with three exact doubling stages: seed would be `3.75` |
| `E_RESHAPE` | Literal projection width `128` cannot supply required `512 * 4 * 4 = 8192` elements |
| `E_AMBIGUOUS` | A convolution inverse leaves several valid input sizes |
| `E_UNRESOLVED` | An inferable argument was never determined by any relation |
| `E_DTYPE` | An `int64` token tensor reaches a port declared in the compute dtype |
| `E_POLICY_CONFLICT` | Literal stride `1` conflicts with the selected doubling policy, or the policy is not declared by that operator |
| `E_ARGUMENT` | An argument is missing, of the wrong type, out of bounds, or supplied twice |
| `E_PRETRAINED` | A checkpoint source is unusable, or the optional pretrained extra is not installed |
| `E_INITIALIZATION` | An initialization constant is invalid or its parameter target is missing or ambiguous |
| `E_TRAINABILITY` | A trainability flag is not a boolean or conflicts across parameter aliases |
| `E_BINDING` | A reference names an undeclared node or tensor port, or the graph contains a cycle |
| `E_NAME` | Undefined/forward local reference or attempted rebinding of a registered operator alias |
| `E_OUTPUT_ARITY` | Tuple unpacking has the wrong number of targets |
| `E_OUTPUT` | Explicit configuration `out` or a non-`None` native return value is not one valid tensor symbol |
| `E_CURRENT` | An omitted tensor input or default output needs current after a multi-output operation cleared it |
| `E_SYNTAX` | Configuration contains an AST form outside `python_config@1` |
| `E_CAPTURE` | Native code mixes capture contexts or coerces a symbolic tensor into an eager value |
| `E_REGISTRY` | An `@operator` declaration is invalid or duplicated, or a module reuses another node's state |
| `E_OPERATOR` | A configuration or capture names an alias the selected registry does not hold |
| `E_SCHEMA` | A contract, saved field, dtype, or device selection is malformed |
| `E_INTEGRITY` | A saved plan's digests or concrete equations do not match its contents |
| `E_RESOURCE` | Source, parser-worker, AST, graph, state-size, or resolution budget is exceeded |
| `E_STATE_VERSION` | Saved state requires an unavailable compatible operator implementation or plan version |
| `E_RUNTIME` | A runtime tensor violates its declared shape, dtype, or device, or the device is unavailable |

Configuration failures report original source line/column locations through the dedent map. Native capture locations are best effort and may use call-site information, but must always identify the affected node/field without requiring function-source inspection. `print(model)` and `repr(model)` must include every layer, its ID/operator, and complete input/output shapes without executing the network. `print(plan)` exposes the equivalent named-port table without a backend. `plan.describe()` must additionally expose provenance sufficiently to explain why a field changed between separately resolved specifications.

Static resolution, dry-run/meta checks, and numerical preflight are distinct inspection modes. Numerical preflight must use disposable modules or restore affected state/RNG; it must not advance a live training stream or alter persistent buffers.

## 14. Acceptance criteria

Each gate is required evidence for the contract above; IMPLEMENTATION.md records which are met by the current release and what remains.

| Gate | Required evidence |
| --- | --- |
| Allocation-free core | Interpret bounded configuration, resolve captured graphs, and serialize while constructing no module and allocating no tensor; malformed specs and unavailable operators fail clearly |
| Shape resolution | Omitted output widths/channels and reshape prefixes; rank from prefix bounds and connected contracts, including bare reshape using upstream element count; ambiguous rank/factorization rejected; generator targets `32×32`, `64×64`, `32×64`; invalid `30×30`; literal-width conflict; inverse ambiguity; group divisibility and join conflicts |
| Ranks and dtypes | `[B, T, D]` sequences resolve and execute through linear/normalization/activation; `float16` and `bfloat16` plans build and round-trip; integer input contracts reach integer ports and are rejected at floating ports with `E_DTYPE` |
| Determinism | Repeated resolution of the same captured graph is identical; equivalent configuration/native/internal graphs with identical node identities and declarations have identical semantic plans; exact alias bindings and policy-order independence |
| Configuration | AST allowlist rejects executable Python outside the subset; locals rebind without mutating edges; unpacking evaluates RHS once; nested calls follow evaluation order; names/comments do not rename nodes; `name` metadata does; optional explicit `out` overrides current and never silently falls back |
| Current semantics | Call-only sequences; positional/keyword tensor disambiguation; assignments do not select current; nested argument evaluation before default binding; explicit-input and join updates; split clearing/recovery; invalid explicit output rejection; empty identity contract checks |
| Loading boundary | Bounded UTF-8 file reads, dedent/source mapping, isolated parse/validation limits, bounded literals, no eval/exec/import dispatch, no fallback between string/file/callable modes |
| Native capture | Trusted fn called exactly once with symbolic input; explicit symbol return or None/fallthrough current selection; helpers/static loops work; context cleanup/concurrency; cross-capture/coercion failures; no callback execution during forward/inspection/restore |
| Operator declarations | Every declared operator has a summary, docstring, category, help text on every argument, and at least one example; every example resolves identically in both frontends, round-trips through JSON, builds, and runs on every available device; operators declaring `reference` match their handwritten equivalent; the generated catalog is current (`python -m hndl.docs --check`) |
| Backend equivalence | Handwritten PyTorch comparison from identical state: forward values, input/parameter gradients, optimizer updates |
| Public sequence API | Tensor forward result; complete shape table from `print(model)` without allocation/RNG draws; integer/negative/name lookup and iteration; shared-module slices; stable state keys without double registration; structural assignment rejected |
| Graph authoring | Single-input current propagation and explicit multi-input ports; branches/joins execute each node once; old tensor references survive rebinding; selected output receives its contract; unused ports allowed but dead nodes fail; node lookup only for branched models |
| Named external ports | Both frontends declare several named inputs and outputs with one shared batch symbol and per-input dtypes; each declared output receives its contract; unused inputs, unselected outputs, alias collisions, and batch disagreement fail with their codes; multi-port plans round-trip through JSON while a single `x` -> `output` plan keeps its previous encoding and digests; runtime binding by position and keyword, and dictionary results, execute and differentiate |
| Split | `128 → 64+64` and `160 → 64+96`; inverse size inference; omitted ambiguous size rejected; non-batch positive axis and nonempty sections validated; both outputs preserve autograd |
| Custom operators | A registry-declared operator resolves through bidirectional shape equality; the DSL's shared symbols, `2*C` scales, literals, ellipsis, variadic inputs, and port dtypes each constrain in both directions; a relation function refines and fails with stable codes; resolution constructs no module; exact identity and version survive persistence |
| Custom graph | Resolve `2*C`; fan-out works; feature/style gradients are correct; reject incorrect batch/channel contracts |
| Numerical validity | Finite affine-normalization forward/backward for constant/nonconstant inputs; first/second derivative checks on suitable nondegenerate fixtures; `dropout` mask statistics and rescaling for `p > 0`, identity in eval mode |
| Pretrained | Text and vision checkpoints resolve their input contract from `config.json`, select each documented output, stay frozen and in eval mode under `model.train()`, unfreeze under `trainable=True`, build in reduced precision, and fail restoration with `E_CONSTRAINT` when the source changes |
| Pretrained providers | A local `.pth` source loads only through a registry-registered builder, requires and verifies `sha256`, returns the `layer=` submodule's output equal to a manual forward hook, lists available submodules for an unknown layer, freezes its parameters, fails restoration when the file changed, and keeps the meta-device parameter probe off the file |
| Pretrained readouts | A `readout=` name resolves only to a callable registered for that provider, returns exactly that callable's tensor, has its output shape traced on the meta device, conflicts with `layer=`, is rejected on transformers and timm sources, fails when it returns more than one tensor, and survives a plan round trip |
| State bounds | The meta pass rejects a plan exceeding `max_state_bytes` before allocating; the build receipt's `state_bytes` matches the constructed model |
| Registration and recovery | Stable registered state; fresh-process round-trip; no first-forward parameters or hidden runtime RNG draws outside `dropout` |
| Initialization | Default construction follows caller RNG and `torch.manual_seed`; explicit construction seeds isolate/restore RNG; inspection consumes no RNG |
| CUDA | Selected operators and the custom fixture execute and reload on CUDA within declared tolerances, in each supported compute dtype |
| Compatibility | Load compatible saved plans without replanning; reject incompatible semantics/state versions actionably |

No identity substitutes, blanket skips, fabricated successful output, or silent device fallback can satisfy these gates. CPU correctness does not establish CUDA correctness. Host integration, complete checkpoint recovery, two-GPU behavior, and multi-host behavior are separate integration gates.

New operators enter through the declaration contract in §8: one module file with its `@operator` declaration, one test module covering its numerics against a reference and its error codes, and a regenerated catalog page. Nothing shared is edited to add an operation, so the operator surface grows without widening the resolver or the frontends.

## 15. Decisions to settle before dependent features ship

The baseline deliberately leaves these details visible:

- Publish machine-readable plan schemas, serialization APIs, and cross-version compatibility fixtures. The public frontend signatures, local-binding semantics, and the `@operator` declaration contract are fixed above; the JSON example illustrates the internal graph model.
- Finalize the canonical JSON encoding's golden digest vectors and the published compatibility matrix for saved plans.
- Complete full derivation chains and source provenance for every inferred value; today each argument records its origin but not the contributing relation.
- Extend initialization beyond constant overrides, and general construction policies beyond per-operator requirement profiles; complete the build-receipt representation and runtime-check controls.
- Define opaque **asserted** contracts for modules that cannot supply relations, including the runtime checks that must accompany them and how backward inference is reported as unavailable.
- Qualify the supported platform, PyTorch version, and device/dtype matrix with numerical tolerances, and make parser isolation configurable beyond the Linux worker.
- Decide whether explicit weight tying between graph nodes enters a later version, and what identity/persistence rules it would require.

These are implementation interface decisions. They do not permit relaxing literal constraints, hiding architecture choices, or treating unresolved shapes as a successful plan.
