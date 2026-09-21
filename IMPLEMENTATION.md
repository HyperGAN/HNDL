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
- Separate, versioned registration of unary shape-preserving operations and
  trusted PyTorch constructors, including the README's SiLU example.
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

All unary operations take an optional leading tensor or `x=`. The remaining
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

Schema 1 uses sorted JSON object keys, compact separators, UTF-8 without ASCII
escaping, arrays for tuples, finite numbers, and SHA-256 digests. Node order
and identities participate in the semantic digest. Source/frontend metadata
and argument provenance affect the artifact digest but not the semantic
digest. Plan JSON is limited to 16 MiB. This is an alpha persistence API;
restoring arbitrary third-party artifacts is not the same isolation boundary
as loading declarative source.

## Remaining v1 work

- General custom operators with explicit argument/port schemas and custom
  bidirectional shape relations; the adaptive instance-normalization fixture.
- General custom construction policies, initializer overrides, and persisted
  per-node trainability masks.
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
