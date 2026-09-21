# Initial alpha: 0.1.0a1

This release establishes the standalone package and basic APIs. `SPEC.md`
remains the broader v1 target, including requirements that are not implemented
in this alpha.

## Available

- Declarative source and bounded UTF-8 file loading: `resolve`, `resolve_file`,
  `network`, and `network_file`.
- Trusted Python authoring: `ops`, `resolve_callable`, and
  `network_from_callable`, with one capture per construction.
- Implicit current tensor, explicit tensor arguments, assignments, comments,
  names, split/remainder, branches, and explicit joins.
- Bidirectional inference for linear widths, reshape products, convolution
  dimensions, split/concat extents, and the exact-doubling `up2` policy.
  Contradictions fail; the resolver does not choose arbitrary missing sizes.
- PyTorch modules, shape tables, name lookup, sequence indexing and shared
  slices, ordinary autograd, state dictionaries, and CPU/CUDA construction.
- Separate, versioned registration of custom operations and trusted PyTorch
  constructors: scalar argument schemas, ordered input/output ports, and
  bidirectional shape rules with shared or integer-scaled dimensions. The
  unary `preserves_shape` helper remains available.
- An executable AdaIN-style example with inferred feature/style projections,
  split/remainder routing, shared branches, and numerical/gradient checks.
- Persisted constant parameter initialization and trainability overrides, shared
  by both frontends and applied when building or rebuilding a plan.
- Immutable resolved graph data, canonical JSON plan persistence, digests,
  exact operator/state-version checks, and `build(plan)` without recapturing an
  author function. JSON is a persistence format, not an authoring language.
- Optional PyTorch dependency, MIT license, wheel/source builds, CI, and a
  release publishing workflow.

## Runtime and operation arguments

The initial supported platform is Linux, on Python 3.11–3.14. Config parsing
uses a Python 3.11 grammar with an explicit AST allowlist in an isolated Linux
worker. Unsupported platforms fail instead of falling back to in-process
parsing. Native authoring executes trusted Python; it is not sandboxed.

The backend requires PyTorch 2.6 or newer in the 2.x series. Plans use
`float32`, with rank-two `BF` or rank-four `NCHW` shapes. Only the batch axis
may be symbolic (`"B"`). Other dimensions and runtime batch sizes must be
positive integers. The device is always caller-selected.

Built-in unary operations take an optional leading tensor or `x=`. Custom
unary operations use their declared input-port keyword. The built-in scalar
arguments are listed below. Omitted inferable fields have no numeric default.

| Call | Arguments and defaults |
| --- | --- |
| `linear()` | Inferable `out_features`; `bias=True` |
| `reshape(...)` | Non-batch shape prefix, positionally or `shape=(...)`; remaining dimensions inferred |
| `flatten()` | Flatten all non-batch axes |
| `relu()`, `tanh()` | No arguments; explicit, out-of-place activations |
| `leaky_relu()` | `negative_slope=0.01` |
| `conv()` | Inferable `out_channels`; required `kernel_size`; `stride=1`, `padding=0`, `dilation=1`, `groups=1`, `bias=True` |
| `deconv()` | Same convolution arguments, plus `output_padding=0`; `policy="up2"` supplies the required doubling settings |
| `group_norm(num_groups)` | Explicit positive group count; `eps=1e-5`, `affine=True`; channels inferred |
| `split()` | Inferable first-section `size`; `dim=1`; returns first section and remainder |
| `add(a, b)` | Both tensors explicit; exact shape equality, no broadcasting |
| `concat(a, b, ...)` | At least two explicit tensors; `axis=1` |

Convolution spatial arguments accept an integer or a pair of integers.
Split/concat axes are positive, non-batch indices; negative axes are rejected.
`name=` sets a stable node ID independently of Python variable names.

## Loading and allocation limits

Both string and file loading use the same parser worker. The fixed parser
limits are 64 KiB of UTF-8 source, 4,096 lines, 16,384 AST nodes, nesting depth
96, 1,024 items per literal container, 16 KiB per string literal, and 256 bits
per integer literal. Worker address space is capped at 256 MiB and CPU time
at two seconds; the parent also enforces a wall-clock timeout. The worker
protocol is bounded at 2 MiB. Parsing never invokes `eval`, `exec`, or a
configuration-provided import or callback.

The resolver and builder additionally bound graph size, dimensions, element
counts, and registered parameter/buffer storage before constructing modules.
Defaults are 4,096 nodes, 16,384 input edges, 1,048,576 per dimension,
268,435,456 elements per example, 1 GiB of registered state, and 256 solver
iterations. Pass a `limits` dictionary to resolution or construction to
override these limits explicitly; keys are `max_nodes`, `max_edges`,
`max_dimension`, `max_elements`, `max_state_bytes`, and `max_iterations`.
Custom implementations are trusted and must honor their declared storage
bounds. These bounds do not estimate peak training memory or bound arbitrary
native Python authoring code.

## Custom operation scope

`Argument`, `Dim`, and `ShapeRule` are pure declarations; they do not import
PyTorch or execute callbacks. Shared dimension names impose equality within
one node, and `Dim("C", scale=2)` imposes an exact integer multiple in either
direction. Literal dimensions are also supported. Patterns include batch and
use the same rank-two/rank-four contracts as built-in operations. See the
[registration examples](README.md#register-your-own-operation) and the
[technical contract](SPEC.md#declarative-custom-schemas-and-shape-rules).

Each registration permits at most 32 input ports, 32 output ports, 64 scalar
arguments, and 64 distinct dimension symbols. Port, argument, and dimension
names are limited to 64 characters. Dimension literals and scales are capped
at `2**31 - 1`; the usually lower runtime `max_dimension` still applies to
resolved extents. Integer scalar arguments are bounded by `abs(value) <=
2**63 - 1`, and string arguments by 16,384 UTF-8 bytes. Float values must be
finite. Schemas may apply tighter numeric bounds. Omitted scalar arguments
need an explicit default; custom scalar inference is not implemented.

The [adaptive-normalization example](examples/adaptive_normalization.py)
shows registration and both a shared mapping branch and split/remainder
routing. Run it with `python examples/adaptive_normalization.py --device cpu`
(or an available CUDA device). Its style-affine zero initialization is an
declaration in the config (`init={"weight": 0, "bias": 0}`); save the state
dictionary along with the plan to preserve trained values.

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
saved concrete equations, dimensions, bounds, and versions without executing
source or an author function. The ordinary PyTorch state dictionary does not
contain the architecture; keep both together.

Schema 2 uses sorted JSON object keys, compact separators, UTF-8 without ASCII
escaping, arrays for tuples, finite numbers, and SHA-256 digests. Node order,
identities, initialization, and trainability participate in the semantic digest. Source/frontend metadata
and argument provenance affect the artifact digest but not the semantic
digest. Plan JSON is limited to 16 MiB. This is an alpha persistence API;
restoring arbitrary third-party artifacts is not the same isolation boundary
as loading declarative source.

The current reader requires plan schema 2 and resolution version 1. Schema 1
plans are rejected with `E_STATE_VERSION`; re-resolve the original definition
and save a new plan. This alpha change adds explicit construction settings to
every node, including defaults. It does not change PyTorch state names.

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
per path. Buffers and nonexistent targets are rejected at build time. Pure
resolution validates the settings without importing or constructing PyTorch
modules; it cannot confirm a trusted module's actual parameter names.
Constants must be numbers other than booleans and must round to finite
float32 values; the plan stores those rounded values. Underflow rounds to zero, and the sign
of zero is preserved. Trainability values must be booleans.
Aliased parameters cannot receive conflicting declarations. Constant targets
must materialize as float32, and cannot share storage with a distinct parameter
or a registered buffer. Trainability changes touching shared storage require
consistent effective flags across the parameters using it.

Initialization overrides run under `no_grad` after parameters are materialized.
Normal constructor initialization still consumes RNG draws even for overridden
parameters. `initialization_seed=None` uses the caller's RNG; an explicit seed
isolates construction and restores the caller's RNG afterward. Constant
application does not draw random values. Freezing affects parameter gradients,
not input gradients or train/eval mode. Host changes to parameters or their
`requires_grad` flags do not rewrite the immutable plan.

## Remaining v1 work

- Custom shape relations beyond equality and integer scaling, inferable custom
  scalar arguments, argument-dependent shapes and state bounds, and opaque
  asserted contracts. Arbitrary custom shape callbacks are not accepted.
- General custom construction policies and initializers beyond constructor
  defaults plus constant parameter overrides.
- Full derivation chains and source provenance for every inferred value.
- Complete persisted author/source metadata, published machine-readable
  schemas and cross-version compatibility fixtures, and strict checkpoint
  compatibility tooling beyond ordinary PyTorch state loading.
- Broader platform qualification, configurable parser isolation, and the full
  numerical/custom-extension acceptance matrix in the spec.

Alpha plan formats may change with an explicit schema version change; this
release does not promise the full v1 persistence or checkpoint contract.
Training loops, metric optimization, optimizer state, and complete experiment
recovery remain owned by the application.

## Development and publication

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and publishing setup.
CI tests a torch-free installation and the CPU backend on Python 3.11 and
3.14, checks the built distributions, and exercises a wheel installation
outside the checkout. CUDA tests execute when hardware is present; CPU CI
does not qualify CUDA execution by itself.
