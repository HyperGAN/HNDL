# Implementation notes: 0.1.2

This release implements the contract in [SPEC.md](SPEC.md). [README.md](README.md)
introduces the APIs with examples; the [operator catalog](docs/operators/index.md)
and the [authored networks](docs/networks.md) are generated from the code.

## Available

- Declarative source and bounded UTF-8 file loading: `resolve`, `resolve_file`,
  `network`, and `network_file`.
- Trusted Python authoring: `ops`, `resolve_callable`, and
  `network_from_callable`, with one capture per construction.
- Implicit current tensor, explicit tensor arguments, assignments, comments,
  names, split/remainder, branches, and explicit joins.
- Bidirectional inference for widths, reshape products, convolution and pooling
  arithmetic, split/concat extents, sequence lengths, and the exact `up2` and
  `down2` policies. Contradictions fail; the resolver does not choose arbitrary
  missing sizes.
- One `@operator` declaration per operation, on the `nn.Module` that
  implements it: alias and version, a shape relation written as
  `x[B, C, H, W], params[B, 2*C] -> out[B, C, H, W]` (shared symbols, integer
  scales, literals, `...` for shared middle axes, `x*` variadic inputs, `out*`
  variadic outputs counted from a declared sequence argument, port dtypes) or a relation function over a bounded node view, scalar arguments
  with help text, runnable examples, and hooks for positional rest arguments,
  policies, validation, and finalization. Built-ins are discovered from
  `hndl.operators`; custom operators use the same decorator on a `Registry`.
  See [docs/ADDING_OPERATORS.md](docs/ADDING_OPERATORS.md).
- Fifty-nine operators across activation, arithmetic, shape, normalization,
  convolution, spatial, sequence, vision, memory, regularization, and
  pretrained categories, each with tests against PyTorch references.
- `pretrained(source)`: a generic loader for transformers and timm checkpoints
  from local paths or `hf://` repositories. See [docs/pretrained.md](docs/pretrained.md).
- PyTorch modules, shape tables, name lookup, sequence indexing and shared
  slices, ordinary autograd, state dictionaries, and CPU/CUDA construction in
  float32, float16, or bfloat16.
- Persisted constant parameter initialization and trainability overrides,
  shared by both frontends and applied when building or rebuilding a plan.
- Immutable resolved graph data, canonical JSON plan persistence, digests,
  exact operator-version checks, and `build(plan)` without recapturing an
  author function. JSON is a persistence format, not an authoring language.
- Generated documentation (`python -m hndl.docs`, checked in CI), MIT license,
  wheel/source builds, CI, and a release publishing workflow.

## Runtime and operation arguments

The supported platform is Linux, on Python 3.11–3.14. Config parsing uses a
Python 3.11 grammar with an explicit AST allowlist in an isolated Linux worker.
Unsupported platforms fail instead of falling back to in-process parsing.
Native authoring executes trusted Python; it is not sandboxed.

The backend requires PyTorch 2.6 or newer in the 2.x series. Tensors have
rank two `[B, F]`, rank three `[B, T, D]` (a sequence of `T` positions with
`D` features; `linear`, normalizations, and activations act on the last axis),
or rank four `[B, C, H, W]`. One-dimensional convolution and pooling use rank
three as `[B, C, L]`; `transpose(1, 2)` moves between the conventions. Only
the batch axis may be symbolic (`"B"`). Other dimensions and runtime batch
sizes must be positive integers. `concat(..., axis=0)` stacks examples, so an
intermediate contract may hold several batches at once, written `"2*B"`;
`chunk(..., dim=0)` divides them again. External contracts stay one plan
batch. The device is always caller-selected.

Plans carry a compute `dtype` of `float32` (default), `float16`, or
`bfloat16`; parameters are constructed in that dtype and every floating tensor
port must match it at runtime. Operators can declare integer ports, such as
`embedding`'s `ids[B, T]:int64` input; pass `input_dtype="int64"` when the
graph input is integer. Edge dtypes are checked at resolution (`E_DTYPE`), so
an integer tensor cannot reach a floating-point port. Reduced precision is
qualified on CUDA.

Built-in unary operations take an optional leading tensor or `x=`. Custom
unary operations use their declared input-port keyword. Every operator's
arguments, defaults, bounds, shape relation, and examples are listed in
[docs/operators](docs/operators/index.md). Convolution spatial arguments
accept an integer or a pair of integers. `split` axes are positive, non-batch
indices; `concat` and `chunk` also accept axis 0, which joins or divides the
batch. Negative axes are rejected. `name=` sets a stable node ID
independently of Python variable names. Operator aliases are reserved names
in configurations.

`dropout` is stochastic in training mode using PyTorch's global RNG, the one
documented deviation from the rule that forward passes draw no hidden
randomness. `pretrained` reads its checkpoint's `config.json` (downloading it
for `hf://` sources) and traces the architecture on the meta device during
resolution, the one documented exception to allocation-free resolution.

## Loading and allocation limits

Both string and file loading use the same parser worker. The fixed parser
limits are 64 KiB of UTF-8 source, 4,096 lines, 16,384 AST nodes, nesting depth
96, 1,024 items per literal container, 16 KiB per string literal, and 256 bits
per integer literal. Worker address space is capped at 256 MiB and CPU time
at two seconds; the parent also enforces a wall-clock timeout. The worker
protocol is bounded at 2 MiB. Parsing never invokes `eval`, `exec`, or a
configuration-provided import or callback.

The resolver and builder additionally bound graph size, dimensions, element
counts, and registered parameter/buffer storage. Defaults are 4,096 nodes,
16,384 input edges, 1,048,576 per dimension, 268,435,456 elements per
example, 1 GiB of registered state, and 256 solver iterations. Pass a
`limits` dictionary to resolution or construction to override these limits
explicitly; keys are `max_nodes`, `max_edges`, `max_dimension`,
`max_elements`, `max_state_bytes`, and `max_iterations`. Storage is measured
by constructing every module on PyTorch's allocation-free `meta` device before
real construction; `build_receipt["state_bytes"]` records the total. These
bounds do not estimate peak training memory or bound arbitrary native Python
authoring code.

## Custom operation scope

`@operator` declarations are data: registration parses the shape string and
argument schema without constructing modules. Shared dimension names impose
equality within one node, `2*C` imposes an exact integer multiple in either
direction, literals fix an axis, and `...` shares a run of middle axes across
the ports that use it. Rules the string cannot express use a `relation`
function over a bounded node view; relations are trusted code. See
[docs/ADDING_OPERATORS.md](docs/ADDING_OPERATORS.md) and the
[technical contract](SPEC.md#8-custom-operators-and-minimal-graphs).

Each declaration permits at most 32 input ports, 32 output ports, 64 scalar
arguments, and 64 distinct dimension symbols. Port, argument, and dimension
names are limited to 64 characters. Dimension literals and scales are capped
at `2**31 - 1`; the usually lower runtime `max_dimension` still applies to
resolved extents. Integer scalar arguments are bounded by `abs(value) <=
2**63 - 1`, and string arguments by 16,384 UTF-8 bytes. Float values must be
finite. Schemas may apply tighter numeric bounds. Omitted scalar arguments
need an explicit default unless marked `inferable`, in which case a shape
symbol or relation must determine them.

## Saving a resolved plan

Use the plan alongside the model's ordinary PyTorch state dictionary:

```python
from pathlib import Path
from hndl import ResolvedPlan
from hndl.torch import build

Path("architecture.json").write_text(model.plan.to_json(), encoding="utf-8")
plan = ResolvedPlan.from_json(
    Path("architecture.json").read_text(encoding="utf-8")
)
restored = build(plan, device="cpu")
restored.load_state_dict(model.state_dict())
# build() returns the lower-level named-input/dictionary-output module:
result = restored(x=batch)["output"]
```

Custom plans require the matching explicit registry at restore/build time.
Saved data cannot import its own implementations. Restoration verifies the
saved concrete equations, dimensions, and operator versions without executing
source or an author function. The ordinary PyTorch state dictionary does not
contain the architecture; keep both together. Plans that use `pretrained`
record the resolved checkpoint revision and fail to restore if the source has
changed.

Schema 1 uses sorted JSON object keys, compact separators, UTF-8 without ASCII
escaping, arrays for tuples, finite numbers, and SHA-256 digests. Node order,
identities, arguments, shapes, dtypes, initialization, and trainability
participate in the semantic digest. Source/frontend metadata and argument
provenance affect the artifact digest but not the semantic digest. Plan JSON
is limited to 16 MiB. Restoring arbitrary third-party artifacts is not the
same isolation boundary as loading declarative source.

## Construction settings

`init={"weight": 0, "bias": 0}` selects constant parameter overrides after
normal module construction. `trainable=False` freezes all parameters in the
operation, while `trainable={"weight": False}` changes only named parameters.
Omitting either option preserves the constructor's values or `requires_grad`
flags. Empty mappings have the same meaning as omission. Explicit `None` is
not supported. Both options are reserved frontend metadata, alongside `name`
and `policy`, and are never passed as operator constructor arguments.

Targets are exact, relative parameter paths (including nested paths such as
`projection.weight`), with at most 256 entries per mapping and 256 characters
per path. Buffers and nonexistent targets are rejected at build time.
Constants must be numbers other than booleans and must round to finite
float32 values; the plan stores those rounded values and fills them into the
parameter's dtype. Trainability values must be booleans. Aliased parameters
cannot receive conflicting declarations. Constant targets cannot share storage
with a distinct parameter or a registered buffer.

Initialization overrides run under `no_grad` after parameters are materialized.
`initialization_seed=None` uses the caller's RNG; an explicit seed isolates
construction and restores the caller's RNG afterward. Freezing affects
parameter gradients, not input gradients or train/eval mode.

## Limits of this release

- Custom shape relations beyond the DSL are trusted Python functions; there is
  no declarative form for arithmetic relations, and inferable custom scalars
  must be bound to a shape symbol or set by a relation.
- Weight tying between graph nodes is unsupported; `pretrained` loads tied
  checkpoints correctly because the wrapped model ties them internally.
- Vision checkpoints load at their native resolution only.
- Full derivation chains for every inferred value, published machine-readable
  plan schemas, and strict checkpoint-compatibility tooling beyond ordinary
  PyTorch state loading remain future work.
- Training loops, metric optimization, optimizer state, and complete experiment
  recovery remain owned by the application.

## Development and publication

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and publishing setup.
CI tests the CPU backend on Python 3.11 and 3.14, checks that the generated
documentation is current, checks the built distributions, and exercises a
wheel installation outside the checkout. CUDA and network-dependent tests run
locally; CPU CI does not qualify CUDA execution or checkpoint downloads.
