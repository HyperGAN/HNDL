# HNDL v1: smart network definition and construction

Status: proposed implementation specification, 2026-09-20. **HNDL**, pronounced “handle,” is the owner-selected name. The owner will create `HyperGAN/hndl`; a coding agent will implement it in a later session. No package, API or syntax in this document is implemented.

HNDL describes partially specified neural networks, resolves their dimensions against input/output contracts and explicit construction policies, and builds ordinary PyTorch modules. Its distinguishing feature is **explainable architecture completion**: a user changes a target image size, and the system resolves compatible seed shapes and projection widths or explains why the constraints cannot be satisfied.

This is an independent package inspired by the useful ideas in HyperGAN 1, with a new implementation and contract. It has no dependency on HyperGAN, its legacy configurable component, Hyperchamber, or the historical layer grammar. HyperGAN is an intended consumer. The [research report](../reports/network-dsl-2026-09-20.md) records the background and standards review.

## 1. Name and scope

The owner's proposal, **HyperGAN Network Definition Language**, gives HNDL a clear origin. For an independent project, the suggested public expansion is **Human-readable Network Definition Language**. The selected name is HNDL and the intended repository is `HyperGAN/hndl`, with package/import name `hndl`. The expanded wording can be settled in its README; it does not affect the design or create a HyperGAN runtime dependency.

Alternatives considered:

| Name | Meaning and assessment |
| --- | --- |
| **HNDL** | Selected by the owner; concise, pronounceable, and describes the authoring interface |
| ShapeLoom | Emphasizes shapes and composition; less explicit about networks |
| LayerWeft | Emphasizes assembling layers; distinctive but “weft” is less familiar |
| TensorPlan | Describes resolution into a concrete plan, but already occurs in research/software naming |

On 2026-09-20, direct PyPI JSON lookups for `hndl`, `hndl-torch`, `shapeloom`, `layerweft` and `tensorplan` returned HTTP 404. This records absence of those project records at lookup time, not a reservation. The HNDL acronym already has unrelated uses, including [hndl-detect](https://pypi.org/project/hndl-detect/); ShapeLoom also has unrelated product/studio uses. TensorPlan is an existing [research name](https://arxiv.org/abs/2110.02195). Naming does not block the design.

### v1 delivers

- A typed Python API and a JSON-compatible structured specification.
- Smart sequential construction with forward shape propagation and bounded backward constraint resolution.
- Versioned policies that supply declared defaults or complete explicitly automatic architecture choices.
- A minimal static graph with named tensor ports, fan-out, joins and custom multi-input modules. Sequences lower into this same representation.
- Ordinary registered PyTorch modules, a frozen resolved plan, per-layer shape summaries and useful errors.
- A custom-layer contract sufficient to define an AdaIN-style operation and wire a small style-conditioned graph.
- Reproducible reconstruction from saved plans and explicit semantic/state compatibility versions.

### Later work

Full StyleGAN recipes and numerical reproduction; modulated/demodulated convolution kernels; automatic topology search; recursive/runtime control flow; dynamic image sizes; parameter-sharing syntax; distributed trainers; checkpoint orchestration; export backends; a visual editor; broad operator catalogs.

A compact layer-string frontend follows the structured resolver within v1. It is deliberately limited to sequences. Graphs use structured data/Python first. This modest graph scope extends the research report's initial sequence-only recommendation in response to the owner's AdaIN/style-flow requirement; it does not introduce a general programming language.

## 2. Responsibilities and invariants

| HNDL owns | Host application owns |
| --- | --- |
| Tensor contracts, operator rules and construction policies | Data, preprocessing and the meaning of model inputs/outputs |
| Constraint propagation, plan expansion and diagnostics | Objectives, regularizers, optimizer ownership and training schedules |
| Module construction, registration, state naming and summaries | Optimizers, EMA, runtime RNG streams and full training recovery |
| Declared custom-layer interfaces and state compatibility | Devices/topology, numerical qualification and deployment |

The following are requirements:

1. **Explicit values are constraints.** Never change a literal width, kernel, stride, output activation or tensor binding to make the network fit.
2. **Inference and choices are different.** A uniquely implied size is inferred. A choice among valid architectures requires a named policy or more user information.
3. **Resolution does not train or allocate tensors.** Built-in resolution needs no torch import, CUDA context, random sampling or network access.
4. **A successful plan is concrete.** Every construction dimension and operator argument is resolved. Only the declared batch symbol may remain variable in v1.
5. **Plans explain themselves.** Each resolved value is labelled explicit, operator default, inferred, or policy-selected, with its source/rule.
6. **Runtime uses normal modules.** No layer strings, shape solving or policy selection run during forward calls.
7. **Unknown behavior is visible.** Unsupported rules, unresolved dimensions, invalid bindings and incompatible implementations are errors. Qualification is a separate question from constructibility.

## 3. Data model

### Tensor contracts

A `TensorSpec` contains ordered dimensions including batch, a dtype, and a layout. Initial layouts are `BF` and `NCHW`; dtype is `float32` for qualified v1 execution. A shared batch symbol such as `B` means equality of batch sizes across graph inputs. Other dimensions must resolve to positive integers before building. Symbolic batch values must be positive at execution.

`AUTO` is an explicit unknown, serialized as `{"auto": true}` in numeric fields. Omitted optional parameters take the selected operator version's documented defaults; omitted required parameters fail. The text frontend's `auto` token lowers to `AUTO`. Unknown values are not interchangeable with `None`, `-1`, zero or an arbitrary symbolic expression.

v1 does not infer color space, image value range, activation choice, mixed precision, tensor layout conversion or broadcasting. Hosts retain semantic data contracts. Built-in joins require their stated dimensions exactly; no hidden broadcasting or resizing makes a join succeed.

### Author specification

The graph form contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Syntax/data-layout version; starts at 1 |
| `inputs` | Named external tensor contracts |
| `nodes` | Ordered declarations: stable ID, versioned operator ID, arguments and input-port bindings |
| `outputs` | Named output references and required tensor contracts |
| `policies` | Versioned policy IDs, arguments, and explicit node selections |
| `metadata` | Optional annotations, excluded from numerical identity |

Node/input/output IDs match `[a-z][a-z0-9_]*`. References are `input:<name>` or `node:<id>/<port>`. A built-in unary operator takes port `x` and produces port `out`. Custom operators declare every port and its contract. Output mappings do not rename internal state.

v1 nodes own independent module instances, even when they use the same operator and arguments. Fan-out reuses a **tensor**, not a module invocation or parameter set. Each node executes once per forward. Explicit weight tying and multiple calls to one stateful module are deferred; requests for them fail rather than cloning silently. This still allows a mapping network's output `w` to feed many independently parameterized style blocks.

The graph must be acyclic. Reject missing references, unknown ports, duplicate IDs, cycles, and nodes unreachable from all declared outputs. Multiple outputs are allowed. Flattening Python tuple/dict outputs into named tensor ports is the custom adapter's responsibility; arbitrary Python objects are not graph values in v1.

### Sequence sugar

`Sequence` supplies one external input `x`, one public output `output`, and consecutive unary nodes connected through `out`. It lowers into the same graph schema and resolver. Explicit IDs are required in the structured form. The text frontend may assign deterministic position-based IDs, but must show them in the saved expansion and warn that inserting a layer changes subsequent generated names. Stable user IDs are preferable for long-lived plans.

## 4. Resolution and policy semantics

Resolution proceeds as follows:

1. Validate structure and bind exact operator/policy versions from an explicitly supplied registry.
2. Expand finite policy templates, if any, into explicit nodes; validate their new IDs and edges. v1's required policies do not choose topology.
3. Seed input/output contracts and explicit arguments; apply policy requirements and operator defaults to omitted fields not reserved for policy selection. Create shape variables and operator relations. `AUTO` fields do not take a fallback default merely because inference is unfinished.
4. Propagate forward and backward to a fixed point, recording derivations and detecting contradictions.
5. Let selected policies fill eligible unresolved choices according to their documented deterministic rules; repeat propagation.
6. Fail on ambiguity, a contradiction, an unavailable rule, or any unresolved non-batch dimension. Emit the complete resolved plan otherwise.

Rules may only add/refine facts, never retract a constraint or replace a literal. Stop when no facts change; implementation must bound expansion and detect non-converging custom rules. v1 uses direct equations, finite candidate sets and interval/divisibility reasoning where specified. A general SMT solver is not required. When the supported rules cannot establish a solution, report `unresolved` rather than claiming the architecture is impossible.

A valid but non-unique integer interval is ambiguous. Do not pick its smallest value unless a selected policy explicitly defines that choice. Distinguish ambiguity from a proven contradiction and from unsupported inference.

### Policy contract

A policy has an immutable ID/version, declared applicability, eligible fields, deterministic selection rules and optional validation constraints. It is invoked during resolution only. Its complete effect is stored in the plan. Built-in policies are pure; custom policies are explicitly registered trusted Python, not arbitrary expressions embedded in data.

Each policy must state which effects are **requirements** and which are **defaults**. For defaults, precedence is explicit user value → uniquely inferred value → selected policy choice → applicable operator default. Apply any policy-imposed requirements as constraints before inference. Contradictory requirements fail, even if one came from the user. Two selected policies assigning different values to the same eligible field fail; selection order never decides the winner. Policies cannot change topology, crop, normalize, add activations or inject noise unless their documented expansion explicitly requests those operations.

Initial required policy:

`spatial.up2_transpose@1` applies only to selected `conv_transpose2d` nodes. It requires kernel 4, stride 2, padding 1, dilation 1, output padding 0 and groups 1. It fills omitted/automatic values with these values and rejects contradictory explicit values. Channels and bias remain separate arguments. This profile guarantees exact spatial doubling. It does not claim artifact-free upsampling or good GAN quality.

A later `spatial.up2_resize_conv` policy would be a different explicit expansion with recorded interpolation/convolution settings. v1 must not silently substitute it when the transpose policy cannot meet a target.

### Required equations

| Operator | Relation and limits |
| --- | --- |
| Linear | `BF → BF`; input width is inferred, output width explicit or constrained by downstream use; no implicit flatten |
| Reshape | Preserve batch and per-example element count; exactly one unresolved product factor may be solved directly |
| Conv2d | `floor((I + 2p - d(k-1) - 1)/s + 1)` per spatial axis; inverse may give multiple possible integer inputs |
| ConvTranspose2d | `(I-1)s - 2p + d(k-1) + output_padding + 1`; no unconditional assumption of doubling |
| Elementwise activation | Preserve all dimensions/layout |
| GroupNorm | Preserve shape; explicit positive group count must divide channels |
| Add | Identical input shapes/layout/dtypes; no broadcasting |
| Concat | Sum the declared non-batch axis; other axes must agree |

v1 reshape may also share explicit dimension variables with connected contracts. Multiple unknown factors remain ambiguous unless other relations or a selected policy determine them. Dimension products and policy expansion have bounded sizes to avoid accidental enormous builds; configured limits fail explicitly rather than truncating.

The minimum built-in catalog is `linear`, `reshape`, `flatten`, `conv2d`, `conv_transpose2d`, `relu`, `leaky_relu`, `tanh`, `group_norm`, `add` and `concat`, each at semantic version 1. Flatten preserves batch and combines all other axes; reshape explicitly supplies the new non-batch dimensions. Add takes named ports `a`/`b`; concat declares an input count, binds ports `x0` through `xN` in that order, and specifies a non-batch axis. v1 built-in linear operates on BF tensors; other ranks require an explicit reshape or custom block. Keep other operators as separately admitted extensions.

Convolution equations follow [PyTorch Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html) and [ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html). Their argument restrictions must also be validated; a shape equation alone does not establish a valid operator.

## 5. Primary example: a smart generator

Proposed Python authoring API; exact helper names may be refined before implementation, but the semantics above are normative:

```python
from hndl import AUTO, Layer, Policy, Sequence, TensorSpec, resolve
from hndl.torch import build

spec = Sequence(
    schema_version=1,
    input=TensorSpec(("B", 128), layout="BF", dtype="float32"),
    output=TensorSpec(("B", 3, 32, 32), layout="NCHW", dtype="float32"),
    policies=[Policy("spatial.up2_transpose@1", nodes=["up1", "up2", "up3"])],
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
plan = resolve(spec)  # Pure construction planning; no torch/CUDA needed.
model = build(plan, device="cuda", initialization_seed=7)
# model(x=z) returns {"output": tensor}; z is supplied by the caller.
```

The two unknown seed axes are determined separately by the two target spatial axes and fixed convolution rules; this is not guessing how to factor a flat vector.

| Target output | Seed map | Projection outputs | Result |
| --- | --- | --- | --- |
| `[B,3,32,32]` | `[B,512,4,4]` | 8192 | Resolved |
| `[B,3,64,64]` | `[B,512,8,8]` | 32768 | Resolved |
| `[B,3,32,64]` | `[B,512,4,8]` | 16384 | Resolved independently per axis |
| `[B,3,30,30]` | Would require non-integer axes under this policy | — | Contradiction |
| `[B,3,32,32]`, literal `project.out_features=128` | Requires 8192 elements | — | Contradiction; preserve the literal |

The basic discriminator example uses explicit stride-2 convolutions `3 → 64 → 128 → 256`, flatten, and `linear 1`. At 32×32, flatten resolves to 4096. It returns an unbounded score; there is no automatic sigmoid. The same builder handles G, D, encoders or unrelated networks without knowing their training roles.

### Compact sequence frontend

The above network may later be authored as a list of strings such as:

```text
project: linear auto
project_relu: relu
seed: reshape 512 auto auto
up1: deconv 256 policy=spatial.up2_transpose@1
act1: relu
up2: deconv 128 policy=spatial.up2_transpose@1
act2: relu
up3: deconv 64 policy=spatial.up2_transpose@1
act3: relu
rgb: conv 3 kernel_size=3 stride=1 padding=1
range: tanh
```

Input/output contracts and `schema_version` remain outside the string list. Grammar: optional `id:`, operator alias, documented positional arguments, then named scalar arguments. Allow integer/finite numeric/boolean/`auto` values and registered policy identifiers. Reject unknown fields and duplicated positional/named arguments. `conv`, `deconv`, `linear` and activation aliases map to exact v1 operator IDs; versions are materialized in the plan. Structured data expresses tuple-valued or complex custom arguments. No Python evaluation, arithmetic expressions, nested language, loops or implicit named-tensor context. A string and equivalent structured specification must produce the same numerical plan.

## 6. Named graphs and AdaIN-style extensions

Smart sequences are the first implementation milestone, but the core representation includes named ports from the outset. The required v1 extension demonstration is this small, static style-conditioned graph:

```text
z [B,128] ── mapping MLP ── w [B,256] ── style affine ── params [B,2C]
                               │                              │
                         feature projection                  │
                               │                              │
                         x [B,C,H,W] ── adaptive norm ── features [B,C,H,W]
```

This example can use a linear projection/reshape from `w` for features. A learned constant, per-layer noise and a complete synthesis stack are later extensions. A custom `adaptive_norm@1` receives named ports `x` and `params`, emits `out`, and contributes the constraints:

- `x` and `out`: `[B,C,H,W]`, same dtype/layout.
- `params`: `[B,2*C]`; its batch equals `x`'s batch.
- `style_affine.out_features=AUTO` therefore resolves to `2*C` when feature channels are known.
- The mapping output can fan out to later style affines without recomputing the mapping or duplicating its parameters.

Define the fixture precisely as **learned affine instance normalization**:

```text
mean = mean(x, spatial_axes, keepdim=True)
variance = mean((x - mean)^2, spatial_axes, keepdim=True)
normalized = (x - mean) / sqrt(variance + eps)
[delta_gamma, beta] = split(params, C, axis=features)
out = (1 + delta_gamma)[B,C,1,1] * normalized + beta[B,C,1,1]
```

`eps` is positive, explicit in the resolved plan, and defaults to `1e-5` for this fixture version. Variance uses the population convention (no unbiased correction). Both feature and style paths remain differentiable. Normalization has no running statistics; train/eval does not change this operation. A zero-initialized style affine gives unit scale and zero bias on **normalized features**, not an identity transform of the raw input. That initialization is explicit in the fixture, never implied for arbitrary linear layers.

The original [AdaIN paper](https://arxiv.org/abs/1703.06868) matches content features to style-feature statistics. The [original StyleGAN](https://arxiv.org/abs/1812.04948) uses styles to control synthesis through adaptive normalization. Our named extension fixture defines its own precise affine convention. It is not a faithful StyleGAN implementation or a claim of identical initialization/training. [StyleGAN2](https://arxiv.org/abs/1912.04958) changes the architecture; a future modulated/demodulated convolution is a different operator with separate semantics and tests.

This demonstrates the extension boundary that later permits mapping/synthesis separation, residual and RGB skip joins, explicit style routing, and custom modulation blocks. New stochastic blocks should accept caller-supplied noise tensors through ports so the host can own RNG/recovery. Dynamic style mixing schedules, implicit global `w`, hidden noise draws and training-phase logic do not belong in v1 graph execution.

## 7. Custom operator contract

A registry entry includes:

| Part | Required behavior |
| --- | --- |
| Identity | Namespaced operator ID and semantic version; duplicate registrations fail |
| Argument schema | Types, required/default values, permitted automatic fields and limits |
| Ports | Named input/output tensor contracts and their ordering where relevant |
| Constraints | Pure rules returning relations, refined values and reasons; no module construction |
| Builder | Constructs a normal `nn.Module` from concrete arguments, without hidden globals or network access |
| Invocation adapter | Binds ports to Python arguments and maps results to named tensor ports |
| State | Stable parameter/buffer names; explicit state compatibility version |
| Capability record | Declared/tested dtype/device and gradient behavior, distinguished from host qualification |

Example relation for `adaptive_norm@1`: `params.features = 2 * x.channels`, batch equality, and output shape equality. The resolver's initial relation vocabulary must include positive integers, equality, integer multiplication by a constant, products, divisibility and fixed convolution formulas. Custom code may emit supported relations or narrow values through bounded pure rules; arbitrary symbolic Python execution is not a data-format feature.

Custom rule code is trusted extension code, loaded only from an explicit registry. The base resolver must not discover/import arbitrary entry points or torch builders merely to validate untrusted data. A host's static validation uses its registered pure rules and reports unknown custom operators; executing custom rule providers is an explicit runtime/preflight boundary. The PyTorch builder and shape provider can be separate modules to preserve torch-free validation.

An opaque module may be used through an adapter with fully declared input/output contracts. Mark those contracts **asserted**, rather than statically verified, and require runtime checks before training. No backward inference is available through an opaque operation unless its adapter supplies rules. Unsupported inference is not a successful no-op. Shape/derivative claims for custom code remain the extension author's responsibility and must be exercised by qualification.

## 8. Resolved plans, modules and initialization

`resolve()` returns an immutable `ResolvedPlan` containing:

- Plan schema and resolution-semantics versions; exact operator/policy versions.
- Canonical nodes, edges, public ports, complete concrete arguments, tensor contracts and verification status.
- Expanded initialization/trainability configuration and execution order.
- A resolution trace and original-source locations for diagnostics.
- Separate semantic and artifact digests, implementation/dependency provenance and state-compatibility requirements.

Use canonical JSON with sorted object keys, ordered node lists, normalized numeric/tuple representations, no non-finite values and SHA-256 digests. Publish the exact canonical encoding and test vectors before shipping serialized plans. The semantic digest covers graph structure, state IDs, operator versions, concrete arguments, tensor contracts, initialization, trainability and execution order. It excludes comments, source locations and inference explanations. The artifact digest covers the entire saved plan excluding its own digest field. Policy IDs and traces are preserved for provenance; once all policy effects are expanded, two spellings producing the same concrete numerical plan have the same semantic identity.

`build(plan, *, device, initialization_seed)` returns `GraphModule`, an ordinary `nn.Module`. `device` is explicit; examples and HyperGAN defaults use CUDA, and CPU fixtures opt in. The plan fixes qualified dtype (`float32` initially). Parameters and buffers are fully materialized before optimizer/distributed setup; no lazy first-forward parameter creation. Runtime architecture checks validate bound input dimensions and report mismatches before applying a layer where possible.

`forward(**inputs)` returns a dictionary keyed by public output names, even for one output. Modules register once under `nodes.n_<node_id>`; the fixed prefix avoids collisions with PyTorch module attribute names. Stateless operators have no state, but retain node identities in execution/diagnostics. Operations execute in a stable topological order, with declaration order breaking ties; that order is saved in the plan. Do not detach, clone, convert devices or insert casts silently. Joins preserve normal PyTorch gradient flow. Built-in activations are out-of-place to avoid mutating tensors used by multiple branches.

Initialization is an explicit part of the plan and build receipt. v1 starts with a versioned `torch_default@1` initializer and explicit constant initialization overrides; its actual behavior depends on the recorded supported PyTorch runtime. Additional algorithms require named versions. A supplied initialization seed uses an isolated RNG scope on the requested device and restores the caller's RNG state. Module construction order is stable and recorded; custom builders must obey this contract. Exact random tensors across devices or PyTorch versions are not promised. Matching a historical GAN's initialization requires an explicit, separately tested profile.

`build` returns a training-mode module using PyTorch conventions. Per-node trainable masks are recorded and applied at construction; freezing parameters does not silently select evaluation mode. The host chooses train/eval behavior and must preserve factory-defined masks when integrating. Built-in v1 normalization uses no running statistics. Stateful custom modules declare persistent buffers and own invalidation/recomputation of derived caches after state loads and device moves.

## 9. State, compatibility and host integration

Persist both the author specification and complete resolved plan. Restore from the saved plan, not by running a newer resolver against the old partially specified architecture. Validate the semantic digest, state names/shapes/dtypes and operator/state compatibility before loading. Old semantic versions must retain their meaning or fail with the affected version and a concrete supported-version/migration route. Architecture changes are a new model or explicit weight transfer, not ordinary continuation.

Keep these versions distinct:

| Version/identity | Purpose |
| --- | --- |
| Package version / Git identity | Identify the implementation used |
| Author/plan schema version | Decode the saved structures |
| Operator/policy semantics version | Preserve mathematical and construction meaning |
| Module state compatibility version | Describe known compatible parameter/buffer layouts and behavior |
| Host checkpoint compatibility version | Decide whether complete training continuation is supported |

Package version alone is not a mathematical compatibility proof. HNDL can expose compatibility metadata, but it must not weaken a consuming application's external dependency checks. HyperGAN currently validates external implementation/dependency identities, so HNDL becomes part of that boundary. Conversely, HyperGAN's own release SHA/dirty state remains provenance on runs/attempts/checkpoints, not a reason by itself to reject resume. Keep HyperGAN's explicit checkpoint compatibility version and its configuration/data/runtime/topology/integrity checks.

HyperGAN integration is a separate adapter: supply tensor contracts from explicit recipe I/O, resolve early through a known pure provider, save the plan, build the module, and unwrap/select the required public tensor output. Losses, priors, regularizers, optimizer/EMA state and complete checkpoint recovery remain in HyperGAN. Compatible but unqualified network variants warn; actual incompatibilities fail. Imported custom layers must not inherit a qualification stamp from a structurally similar built-in layer.

CUDA is the product default. CPU CI exercises small correctness fixtures. GPU save/resume, complete two-GPU numerical and failure/recovery behavior, and actual multi-host execution remain distinct HyperGAN gates. HNDL cannot establish distributed training correctness by constructing a graph, exporting it, or passing an NCCL diagnostic.

## 10. Diagnostics and inspection

Each diagnostic has a stable code, source/node/field location, affected constraint chain and suggested explicit remedies. Suggested repairs never apply themselves.

| Code | Example |
| --- | --- |
| `E_CONSTRAINT` | Target height 30 conflicts with three exact doubling stages; an integer seed height would have to be 3.75 |
| `E_RESHAPE` | `project` explicitly produces 128 values but `seed` needs 8192 |
| `E_AMBIGUOUS` | A downsampling inverse admits multiple integer inputs; specify the input size or choose a policy |
| `E_UNRESOLVED` | Custom block lacks a relation needed to infer its input width |
| `E_POLICY_CONFLICT` | `up1.stride=1` contradicts `spatial.up2_transpose@1` |
| `E_BINDING` | `node:style/scale` references an undeclared output port |
| `E_STATE_VERSION` | Saved operator state version is unavailable; identify the required compatible implementation |

`plan.describe()` should show node/port shapes, parameter counts where statically known, inferred/defaulted/policy values, custom asserted contracts and the semantic digest. Parameter count is not an estimate of peak training memory; do not present it as one. Dry-run/meta checks and actual numerical preflight are separate inspection modes with explicit results and limits. Numerical preflight runs on disposable modules or restores all affected state/RNG; it must not advance a live training stream or change normalization buffers.

## 11. Qualification and acceptance

v1 correctness is defined by these gates; this design session has not run them:

| Gate | Required evidence |
| --- | --- |
| Pure core | Import/resolve/serialize without torch or CUDA; malformed specs and unavailable custom providers fail clearly |
| Shape resolution | 32/64/non-square generator targets, incompatible 30 target, explicit-width conflict, convolution inverse ambiguity, group divisibility and graph join conflicts |
| Determinism | Repeated resolution is identical; equivalent sequence/graph/string forms have identical semantic plans; policy order cannot change results |
| Module equivalence | Compare with handwritten PyTorch modules initialized from identical state: forward values, input/parameter gradients and optimizer updates |
| Custom flow | AdaIN-style fixture resolves `2*C`, handles feature/style fan-out, has correct gradients on both paths, and rejects wrong batch/channel contracts |
| Numerical validity | Finite forward/backward at constant and nonconstant AdaIN inputs; meaningful first- and second-derivative checks on small nondegenerate fixtures and applicable GAN penalties |
| Registration and recovery | All parameters/buffers registered; stable state keys; state round-trip in fresh process; no first-forward new parameters or hidden RNG consumption |
| CUDA | Actual selected operators/custom fixture execute and reload on CUDA with declared numerical tolerance; CPU correctness alone is insufficient |
| Compatibility | Load an old saved resolved plan under a compatible new package build without replanning; reject known incompatible operator/state versions actionably |
| HyperGAN adapter | Installed CLI construction, selected tensor output, complete save/resume/earlier-snapshot recovery; separate two-GPU and multi-host qualification |

No blanket skips, identity substitutions, silent CPU fallback or fake successful output may satisfy a gate. Builder correctness does not imply GAN quality, StyleGAN fidelity, arbitrary custom-operator support or exportability.

## 12. Implementation sequence for the next session

1. **Independent core package:** work in the owner-created `HyperGAN/hndl` repository; establish packaging/license, schema/types, registry and resolution trace. Implement linear/reshape/activation equations and pure plan serialization. No HyperGAN imports.
2. **Smart sequence:** convolution rules, `spatial.up2_transpose@1`, the generator/discriminator examples and diagnostics. Prove forward/backward resolution before adding string syntax.
3. **PyTorch builder:** registration, initialization isolation, state reconstruction, CPU equivalence and CUDA execution. Keep backend imports out of the core.
4. **Minimal graph and extension proof:** named ports, fan-out, add/concat, custom provider contract and the AdaIN-style fixture. This is part of v1 acceptance, not a full StyleGAN milestone.
5. **Authoring frontend and docs:** bounded layer strings, stable IDs, equivalence tests, plan summaries and a standalone walkthrough.
6. **HyperGAN adapter:** a small PR targeting `develop`, with resolved-plan persistence and current validation/recovery contracts; no rewrite of the pinned CIFAR recipe merely to demonstrate coverage.

Proposed package structure is `hndl` for pure specifications/resolution and `hndl.torch` for module building, with a `[torch]` installation extra. Operator definitions and shape rules belong to the pure core; torch constructors load only through the backend. Separate package publishing, release numbering and broader operator admission follow actual acceptance rather than this design.

The next session should begin with this document and the research report, then implement the pure resolver slice in the owner-created repository. The basic contract, bounded policy model and custom multi-input route can proceed without deciding every future style operator.

### Coding-agent handoff

Copy this document into the new repository as its initial design reference, retaining the HyperGAN research link/attribution. The research report can be copied alongside it with relative links adjusted or linked at its eventual published revision; do not assume those local links exist in the new checkout. Start with steps 1–2 above and their pure-core/shape/determinism acceptance gates. Resolve any contradictions in the spec explicitly and document chosen public API details before building dependent features.

Keep implementation and package tests in `HyperGAN/hndl`. No legacy source port, HyperGAN dependency, GAN trainer, complete StyleGAN implementation or package publication is part of that initial slice. Record completed gates and the next concrete action in the new repository's own status document. The final v1 still includes the small named-graph/AdaIN extension proof; a sequence-only first slice is an implementation checkpoint, not a claim that the full specification is complete.

## 13. Design provenance and verification

The owner accepted the research direction, requested a v1 specification before implementation, asked to accommodate AdaIN/style-based flows without losing the basic scope, and allowed a fresh implementation independent of HyperGAN 1. The owner then selected HNDL (“handle”) and stated that they will create `HyperGAN/hndl` before starting a coding agent. This document records those directions and proposes the concrete contracts above; it does not claim that every API detail has been accepted.

Local source reviewed: `dd38b02dc70bfdecac97300a5964611e447c8f96`; fetched and GitHub `develop`: `f8f5ae15c61ff48a88bedbe6a6818ebd9e91cde0`, still 22 commits behind local. The earlier research report/status changes and untracked `.claude/` were present on entry. Drafting used the external `/home/martyn/dev/hypergan/smart-networks-v1` worktree, branch `docs/smart-networks-v1`. No legacy code was copied into a new implementation.

Primary sources are linked where used; broader standards context remains in the research report. Naming searches also found the existing [TensorLoom package](https://pypi.org/project/tensorloom/), so that name is excluded. Its project description is not evidence that HNDL's proposed constraint contract already exists or a reason to adopt its runtime. ONNX/NNEF export remains deferred.

Document validation passed: the proposed Python example parses with `ast.parse`, the local research link resolves, code fences balance, and whitespace checks pass. Direct integer checks reproduce the 32×32, 64×64 and 32×64 projection/seed values, the impossible 30×30 target and literal-128 conflict, and D's 4096 flattened features. These are specification checks, not execution of the unimplemented API or acceptance of a numerical runtime.

This is a local documentation deliverable, not a new runtime, dependency installation, repository publication, PR, GPU job or package release. The status ledger records validation results and the next action.
