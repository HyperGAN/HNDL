"""Bounded, monotone tensor-shape resolution without importing torch."""

from dataclasses import replace
from collections.abc import Mapping
from math import prod
import re
import warnings

from .errors import HNDLError
from .operator import COMPUTE_DTYPES, ELLIPSIS, INDEX_DTYPES, NodeView, SUPPORTED_RANKS, Sym
from .registry import Registry, normalize_arguments
from .types import (BATCH, EXTERNAL_INPUT, EXTERNAL_OUTPUT, Graph, Node, ResolvedNode, ResolvedPlan,
                    batch_multiple)


# Each default stops a real failure: an unbounded graph or port count, a typo
# that asks for a huge tensor or model (checked on the meta device before any
# allocation), or a relation that never settles. They are not budgets for
# legitimate networks; pass ``limits=`` to raise one.
DEFAULT_LIMITS = {
    "max_nodes": 4096, "max_edges": 16_384, "max_dimension": 1_048_576,
    "max_elements": 268_435_456, "max_state_bytes": 64 * 2**30,
    "max_iterations": 256,
}
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


def _limits(overrides):
    result = dict(DEFAULT_LIMITS)
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - result.keys():
            raise HNDLError("E_RESOURCE", f"Unknown resource limits; expected {', '.join(result)}")
        result.update(overrides)
    if any(type(value) is not int or value <= 0 for value in result.values()):
        raise HNDLError("E_RESOURCE", "All resource limits must be positive integers")
    return result


def _contract(shape, label, limits, *, multiples=False):
    """Validate one shape tuple. ``multiples`` admits a ``"k*B"`` batch entry.

    External contracts stay one plan batch (``"B"``, or a fixed positive
    integer); only the internal port contracts a batch-axis join or split
    produces may carry a multiple of it.
    """
    if not isinstance(shape, (tuple, list)) or len(shape) not in SUPPORTED_RANKS:
        ranks = ", ".join(map(str, SUPPORTED_RANKS))
        raise HNDLError("E_SCHEMA", f"{label} must include batch and have a supported rank ({ranks})")
    for axis, value in enumerate(shape):
        if axis == 0 and (value == BATCH or (multiples and batch_multiple(value) is not None)):
            continue
        if type(value) is not int or value <= 0:
            allowed = "'B' or 'k*B'" if axis == 0 and multiples else "'B'"
            raise HNDLError("E_SCHEMA", f"{label}[{axis}] must be positive, with only batch allowed to be {allowed}")
        if value > limits["max_dimension"]:
            raise HNDLError("E_RESOURCE", f"{label}[{axis}] exceeds max_dimension")
    if prod(shape[1:]) > limits["max_elements"]:
        raise HNDLError("E_RESOURCE", f"{label} per-example element count exceeds max_elements")
    return tuple(shape)


def _ordered_nodes(graph, registry, limits):
    if len(graph.nodes) > limits["max_nodes"]:
        raise HNDLError("E_RESOURCE", "Graph exceeds max_nodes")
    lookup = {}
    specs = {}
    declarations = {}
    edge_count = 0
    for index, node in enumerate(graph.nodes):
        if not isinstance(node, Node) or not isinstance(node.id, str) or not _ID.fullmatch(node.id):
            raise HNDLError("E_SCHEMA", "Nodes must have IDs matching [a-z][a-z0-9_]*")
        if node.id in lookup:
            raise HNDLError("E_BINDING", f"Duplicate node ID {node.id}")
        edge_count += len(node.inputs)
        if edge_count > limits["max_edges"]:
            raise HNDLError("E_RESOURCE", "Graph exceeds max_edges")
        spec = registry.by_identity(node.op)
        if node.source is not None and not isinstance(node.source, Mapping):
            raise HNDLError("E_SCHEMA", "Node source metadata must be a mapping", node=node.id)
        args = normalize_arguments(spec, (), node.args)
        # A variadic output operator derives its ports from its own arguments.
        output_ports = spec.output_ports_for(args)
        if tuple(node.outputs) != output_ports:
            raise HNDLError("E_BINDING", f"{node.id} output ports must be {output_ports}", node=node.id)
        if spec.variadic is not None and args["input_count"] > limits["max_edges"]:
            raise HNDLError("E_RESOURCE", f"{spec.alias} input_count exceeds max_edges", node=node.id)
        source = dict(node.source or {})
        origins = {key: "explicit" if key in node.args else "operator default" for key in args}
        if "policy" in node.args:
            policy = spec.policies[node.args["policy"]]
            source["policy"] = policy.identity
            for key in policy.requires:
                if key not in node.args:
                    origins[key] = "policy-selected"
        origins.update(source.get("argument_origins", {}))
        source["argument_origins"] = origins
        ports = spec.input_ports_for(args)
        if set(node.inputs) != set(ports):
            raise HNDLError("E_BINDING", f"Required input ports are {ports}, got {tuple(node.inputs)}", node=node.id)
        lookup[node.id] = replace(node, args=args, source=source)
        specs[node.id] = spec
        declarations[node.id] = index
    external = {f"input:{name}" for name in graph.inputs}
    refs = set(external)
    refs.update(f"node:{node.id}/{port}" for node in lookup.values() for port in node.outputs)
    for entry in graph.outputs.values():
        if not isinstance(entry["ref"], str) or entry["ref"] not in refs:
            raise HNDLError("E_BINDING", f"Unknown public output reference {entry['ref']!r}")
    dependencies = {}
    consumers = {node_id: [] for node_id in lookup}
    for node in lookup.values():
        deps = set()
        for ref in node.inputs.values():
            if not isinstance(ref, str) or ref not in refs:
                raise HNDLError("E_BINDING", f"Unknown input reference {ref!r}", node=node.id)
            if ref not in external:
                deps.add(ref[5:].split("/", 1)[0])
        dependencies[node.id] = deps
        for dep in deps:
            consumers[dep].append(node.id)
    ready = [key for key, deps in dependencies.items() if not deps]
    ordered = []
    pending = {key: set(deps) for key, deps in dependencies.items()}
    while ready:
        ready.sort(key=declarations.__getitem__)
        key = ready.pop(0)
        ordered.append(lookup[key])
        for consumer in consumers[key]:
            pending[consumer].remove(key)
            if not pending[consumer]:
                ready.append(consumer)
    if len(ordered) != len(lookup):
        raise HNDLError("E_BINDING", "Graph contains a cycle")
    reachable = set()
    queue = [entry["ref"][5:].split("/", 1)[0] for entry in graph.outputs.values()
             if entry["ref"] not in external]
    while queue:
        key = queue.pop()
        if key not in reachable:
            reachable.add(key)
            queue.extend(dependencies[key])
    dead = set(lookup) - reachable
    if dead:
        raise HNDLError("E_BINDING", f"Nodes do not reach the selected output: {', '.join(sorted(dead))}")
    # Every declared input must reach the graph; a silently ignored contract is
    # an author error, not an implicit optional input.
    used = {ref for node_id in reachable for ref in lookup[node_id].inputs.values()}
    used.update(entry["ref"] for entry in graph.outputs.values())
    unused = [name for name in graph.inputs if f"input:{name}" not in used]
    if unused:
        raise HNDLError("E_BINDING", f"Declared inputs are never used: {', '.join(unused)}")
    # Port dtypes are static per operator; check every edge once.
    dtypes = {f"input:{name}": entry["dtype"] for name, entry in graph.inputs.items()}
    for node in ordered:
        spec = specs[node.id]
        for port, ref in node.inputs.items():
            expected = spec.port_dtype(port, graph.dtype)
            if expected != "any" and dtypes[ref] != expected:
                raise HNDLError("E_DTYPE", f"Port {port} expects {expected} but {ref} carries {dtypes[ref]}", node=node.id)
        for port in node.outputs:
            declared = spec.port_dtype(port, graph.dtype)
            dtypes[f"node:{node.id}/{port}"] = graph.dtype if declared == "any" else declared
    return ordered, specs, dtypes


def _canonical_arguments(spec, args, source):
    """Drop arguments added after release (``Arg(since=...)``) that hold their default.

    Such a node means exactly what it meant before the argument existed, so it
    keeps that release's args, argument origins and therefore digests. Saved
    plans without the argument load because resolution fills the default and
    drops it again; module construction fills it back in.
    """
    omitted = {name for name, value in args.items() if name in spec.args and spec.args[name].omitted(value)}
    if not omitted:
        return args, source
    args = {name: value for name, value in args.items() if name not in omitted}
    if source and "argument_origins" in source:
        source = dict(source)
        source["argument_origins"] = {name: origin for name, origin in source["argument_origins"].items()
                                      if name not in omitted}
    return args, source


class _Solver:
    def __init__(self, graph, nodes, specs, limits, dtypes=None):
        self.graph, self.nodes, self.specs, self.limits = graph, nodes, specs, limits
        self.dtypes = {} if dtypes is None else dtypes
        self.batch = graph.input_shape[0]
        # The batch entry every port of the node being solved shares, unless the
        # operator declares that it moves tensors across the batch axis itself.
        self.node_batch = None
        self.shapes = {f"input:{name}": list(entry["shape"]) for name, entry in graph.inputs.items()}
        self.args = {node.id: dict(node.args) for node in nodes}
        self.changed = False
        self.node = None
        self.intervals = {}
        for entry in graph.outputs.values():
            self.set_shape(entry["ref"], entry["shape"])

    def error(self, code, message):
        source = self.node.source if self.node and self.node.source else {}
        raise HNDLError(code, message, node=self.node.id if self.node else None,
                        line=source.get("line"), column=source.get("column"))

    def set_shape(self, ref, values, code="E_CONSTRAINT"):
        values = list(values)
        if len(values) not in SUPPORTED_RANKS:
            self.error(code, f"Unsupported tensor rank {len(values)} at {ref}")
        for axis, value in enumerate(values):
            if value is None:
                continue
            if axis == 0:
                if not self.valid_batch(value):
                    self.error(code, f"Batch contract {value!r} conflicts with {self.batch!r} at {ref}")
                continue
            if type(value) is not int or value <= 0:
                self.error(code, f"Dimension {axis} at {ref} must be positive, got {value}")
            if value > self.limits["max_dimension"]:
                self.error("E_RESOURCE", f"Dimension {axis} at {ref} exceeds max_dimension")
        known = [value for value in values[1:] if value is not None]
        if prod(known) > self.limits["max_elements"]:
            self.error("E_RESOURCE", f"Tensor {ref} exceeds max_elements per example")
        old = self.shapes.get(ref)
        if old is None:
            self.shapes[ref] = values
            self.changed = True
            return
        if len(old) != len(values):
            self.error(code, f"Rank conflict for {ref}: {len(old)} versus {len(values)}")
        for index, value in enumerate(values):
            if value is None:
                continue
            if old[index] is None:
                old[index] = value
                self.changed = True
            elif old[index] != value:
                if index == 0:
                    self.error(code, f"Batch {old[index]!r} at {ref} conflicts with required {value!r}")
                else:
                    self.error(code, f"Dimension {index} at {ref}: {old[index]} conflicts with required {value}")

    def valid_batch(self, value):
        """Whether a batch entry is admissible for this plan.

        A plan whose contracts fix a concrete batch keeps integer batches; a
        symbolic plan carries ``"B"`` or a multiple of it, which only a
        batch-axis join or split introduces.
        """
        if type(self.batch) is int:
            return type(value) is int and value > 0
        return batch_multiple(value) is not None

    def share_batch(self, refs):
        """Batch passes through unchanged, so one node's ports agree on it.

        The entry may be ``"B"`` or a multiple such as ``"2*B"``; an operator
        that moves tensors across the batch axis declares ``batch="relation"``
        and is excluded here, setting axis 0 on each port itself.
        """
        known = None
        for ref in refs:
            shape = self.shapes.get(ref)
            if shape is None or shape[0] is None:
                continue
            if known is not None and shape[0] != known:
                self.error("E_CONSTRAINT", f"Batch {shape[0]!r} at {ref} conflicts with {known!r} on the same node")
            known = shape[0]
        self.node_batch = known
        if known is None:
            return
        for ref in refs:
            shape = self.shapes.get(ref)
            if shape is not None and shape[0] is None:
                shape[0] = known
                self.changed = True

    def rank(self, ref, rank):
        shape = self.shapes.get(ref)
        batch = self.node_batch if shape is None else shape[0]
        self.set_shape(ref, [batch] + [None] * (rank - 1))
        return self.shapes[ref]

    def axis(self, ref, axis, value, code="E_CONSTRAINT"):
        shape = self.shapes.get(ref)
        if shape is None or value is None:
            return
        updated = [None] * len(shape)
        updated[0] = shape[0]
        updated[axis] = value
        self.set_shape(ref, updated, code)

    def equal(self, left, right):
        a, b = self.shapes.get(left), self.shapes.get(right)
        if a is not None:
            self.set_shape(right, a)
        if b is not None:
            self.set_shape(left, b)

    def argument(self, args, name, value):
        if value is None:
            return
        if name not in args:
            args[name] = value
            self.changed = True
        elif args[name] != value:
            self.error("E_CONSTRAINT", f"Explicit {name}={args[name]} conflicts with required {value}")

    def product(self, left, right):
        a, b = self.shapes.get(left), self.shapes.get(right)
        if a is None or b is None:
            return
        for source, target, ref in ((a, b, right), (b, a, left)):
            if any(value is None for value in source[1:]):
                continue
            total = prod(source[1:])
            missing = [i for i in range(1, len(target)) if target[i] is None]
            known = prod(value for value in target[1:] if value is not None)
            if not missing:
                if known != total:
                    self.error("E_RESHAPE", f"Element count {total} cannot satisfy reshape count {known}; preserve explicit dimensions")
            elif len(missing) == 1:
                if total % known or total // known <= 0:
                    self.error("E_RESHAPE", f"Element count {total} is not divisible by known reshape product {known}")
                self.axis(ref, missing[0], total // known, "E_RESHAPE")

    def interval(self, ref, axis, lower, upper):
        """Intersect a bounded inverse interval, fixing the axis when it collapses."""
        previous = self.intervals.get((ref, axis))
        if previous:
            lower, upper = max(lower, previous[0]), min(upper, previous[1])
        if lower > upper:
            self.error("E_CONSTRAINT", "Convolution inverse intervals are incompatible")
        self.intervals[ref, axis] = (lower, upper)
        shape = self.shapes.get(ref)
        known = shape[axis] if shape is not None and axis < len(shape) else None
        if known is not None and not lower <= known <= upper:
            self.error("E_CONSTRAINT", f"Input extent {known} lies outside required [{lower}, {upper}]")
        if lower == upper:
            self.axis(ref, axis, lower)

    def port_refs(self, node):
        return list(node.inputs.values()) + [f"node:{node.id}/{port}" for port in node.outputs]

    def apply(self, node):
        self.node = node
        spec = self.specs[node.id]
        view = NodeView(self, node, spec)
        refs = None if spec.batch == "relation" else self.port_refs(node)
        self.node_batch = None
        if refs is not None:
            self.share_batch(refs)
        self.patterns(spec, view)
        if spec.relation is not None:
            spec.relation(view)
        # A relation may have created ports after the first pass; give the
        # shared batch to those too rather than waiting for another sweep.
        if refs is not None:
            self.share_batch(refs)

    def patterns(self, spec, view):
        """Refine shared positive symbols from the declared shape patterns.

        Symbols live only in this node invocation. Input and output facts
        participate equally; every shared occurrence is an equation. An
        ellipsis stands for the same run of middle axes on every port using it.
        """
        entries = [(port.name, port.pattern) for port in (*spec.inputs, *spec.outputs) if port.pattern is not None]
        if not entries:
            return
        args = view.args
        bound = {arg.dim: name for name, arg in spec.args.items() if arg.dim is not None}
        # Determine the shared ellipsis length from any port with a known rank.
        ellipsis_length = None
        for name, pattern in entries:
            if ELLIPSIS in pattern:
                shape = view.shape(name)
                if shape is not None:
                    ellipsis_length = len(shape) - (len(pattern) - 1)
                    break
        expanded = []
        for name, pattern in entries:
            if ELLIPSIS in pattern:
                if ellipsis_length is None:
                    continue
                if ellipsis_length < 0:
                    self.error("E_CONSTRAINT", f"Rank of {name} is below the declared pattern rank")
                index = pattern.index(ELLIPSIS)
                pattern = pattern[:index] + tuple(Sym(f"...{i}") for i in range(ellipsis_length)) + pattern[index + 1:]
            expanded.append((name, pattern))
        values = {}
        for symbol, arg_name in bound.items():
            if arg_name in args:
                values[symbol] = args[arg_name]
        for name, pattern in expanded:
            shape = view.rank(name, len(pattern))
            for axis, dimension in enumerate(pattern):
                if axis == 0:
                    continue
                if type(dimension) is int:
                    view.axis(name, axis, dimension)
                    continue
                extent = shape[axis]
                if extent is None:
                    continue
                if extent % dimension.scale:
                    self.error("E_CONSTRAINT", f"Extent {extent} at {name}[{axis}] is not divisible by scale {dimension.scale} of {dimension.name}")
                value = extent // dimension.scale
                if dimension.name in values and values[dimension.name] != value:
                    self.error("E_CONSTRAINT", f"Shared dimension {dimension.name} requires incompatible values {values[dimension.name]} and {value}")
                values[dimension.name] = value
        for name, pattern in expanded:
            for axis, dimension in enumerate(pattern):
                if axis and isinstance(dimension, Sym) and dimension.name in values:
                    view.axis(name, axis, dimension.scale * values[dimension.name])
        for symbol, arg_name in bound.items():
            if symbol in values:
                view.arg(arg_name, values[symbol])

    def run(self):
        for _ in range(self.limits["max_iterations"]):
            self.changed = False
            for node in self.nodes:
                self.apply(node)
            for node in reversed(self.nodes):
                self.apply(node)
            if not self.changed:
                break
        else:
            self.error("E_RESOURCE", "Shape rules exceeded max_iterations")
        resolved = []
        for node in self.nodes:
            self.node = node
            spec = self.specs[node.id]
            args = self.args[node.id]
            for port, ref in list(node.inputs.items()) + [(port, f"node:{node.id}/{port}") for port in node.outputs]:
                shape = self.shapes.get(ref)
                if shape is None or any(value is None for value in shape):
                    interval = next((bounds for (r, _), bounds in self.intervals.items() if r == ref), None)
                    detail = f"; inverse allows {interval}" if interval else ""
                    self.error("E_AMBIGUOUS", f"Shape for {port} ({ref}) remains ambiguous{detail}; supply a dimension or a policy")
            missing = [name for name, arg in spec.args.items() if arg.inferable and name not in args]
            if missing:
                self.error("E_UNRESOLVED", f"{spec.alias} could not infer {', '.join(missing)}")
            input_shapes = {key: tuple(self.shapes[ref]) for key, ref in node.inputs.items()}
            output_shapes = {key: tuple(self.shapes[f"node:{node.id}/{key}"]) for key in node.outputs}
            if spec.finalize is not None:
                args = dict(spec.finalize(dict(args), input_shapes, output_shapes))
            args, source = _canonical_arguments(spec, args, node.source)
            origins = {}
            source_origins = source.get("argument_origins", {}) if source else {}
            for name in args:
                origins[name] = source_origins.get(name, "explicit" if name in node.args else "inferred")
            resolved.append(ResolvedNode(
                id=node.id, op=node.op, args=args, inputs=node.inputs, outputs=node.outputs, source=source,
                input_shapes=input_shapes, output_shapes=output_shapes, provenance=origins,
                initialization=node.initialization, trainability=node.trainability,
            ))
        return tuple(resolved)


def resolve_graph(graph, registry=None, limits=None):
    if not isinstance(graph, Graph):
        raise HNDLError("E_SCHEMA", "resolve_graph requires a captured Graph")
    registry = Registry.builtins() if registry is None else registry
    if not isinstance(registry, Registry):
        raise HNDLError("E_REGISTRY", "registry must be an explicit Registry")
    limits = _limits(limits)
    if graph.dtype not in COMPUTE_DTYPES:
        raise HNDLError("E_SCHEMA", f"dtype must be one of {', '.join(COMPUTE_DTYPES)}")
    if not isinstance(graph.frontend, str) or len(graph.frontend) > 256:
        raise HNDLError("E_SCHEMA", "Frontend provenance must be a bounded string")
    single_input = tuple(graph.inputs) == (EXTERNAL_INPUT,)
    inputs = {}
    for name, entry in graph.inputs.items():
        label = "input_shape" if single_input else f"input_shape[{name}]"
        dtype_label = "input_dtype" if single_input else f"input_dtype[{name}]"
        if entry["dtype"] not in COMPUTE_DTYPES + INDEX_DTYPES:
            raise HNDLError("E_SCHEMA", f"{dtype_label} must be one of {', '.join(COMPUTE_DTYPES + INDEX_DTYPES)}")
        if entry["dtype"] in COMPUTE_DTYPES and entry["dtype"] != graph.dtype:
            raise HNDLError("E_SCHEMA", "A floating-point input_dtype must equal the plan dtype")
        inputs[name] = {"shape": _contract(entry["shape"], label, limits), "dtype": entry["dtype"]}
    single_output = tuple(graph.outputs) == (EXTERNAL_OUTPUT,)
    outputs = {}
    for name, entry in graph.outputs.items():
        label = "output_shape" if single_output else f"output_shape[{name}]"
        outputs[name] = {"ref": entry["ref"], "shape": _contract(entry["shape"], label, limits)}
    # One shared batch symbol ties every external contract together.
    batches = {entry["shape"][0] for entry in (*inputs.values(), *outputs.values())}
    if len(batches) > 1:
        raise HNDLError("E_CONSTRAINT", "Input and output must declare the same batch dimension")
    nodes, specs, dtypes = _ordered_nodes(graph, registry, limits)
    solver = _Solver(graph, nodes, specs, limits, dtypes)
    # Relations that look up host-registered providers see this registry, never a global.
    with registry.activated():
        resolved = solver.run()
    leading_input = next(iter(inputs.values()))
    leading_output = next(iter(outputs.values()))
    return ResolvedPlan(resolved, leading_input["shape"], leading_output["shape"], leading_output["ref"],
                        graph.dtype, graph.frontend, registry, input_dtype=leading_input["dtype"],
                        named_inputs=inputs, named_outputs=outputs)


def _shape_text(shape):
    return "[" + ", ".join(str(part) for part in shape) + "]"


def _plan_differences(plan, verified, registry):
    """Compare a saved plan with its re-resolution, node by node.

    Returns ``(benign, conflicts)`` as readable lines. A difference is benign
    when the missing value is fully determined by what the saved plan already
    fixes: an operator's declared default, or an argument bound to a dimension
    symbol whose extent is read off a saved (and verified) port shape. Anything
    else --- a port shape that does not satisfy the equations, a saved value
    the equations change, or an argument only a policy or relation search
    would choose --- is a conflict, because accepting it would silently change
    the network the plan and its checkpoint describe.
    """
    benign, conflicts = [], []
    resolved = {node.id: node for node in verified.nodes}
    for saved in plan.nodes:
        node = resolved[saved.id]
        spec = registry.by_identity(saved.op)
        for kind, saved_ports, ports in (("input", saved.input_shapes, node.input_shapes),
                                         ("output", saved.output_shapes, node.output_shapes)):
            for port, shape in saved_ports.items():
                if tuple(ports[port]) != tuple(shape):
                    conflicts.append(f"node {saved.id}: {kind} {port} is saved as {_shape_text(shape)} "
                                     f"but the equations give {_shape_text(ports[port])}")
        for name in sorted(set(saved.args) | set(node.args)):
            if name in saved.args and name in node.args:
                if saved.args[name] != node.args[name]:
                    conflicts.append(f"node {saved.id}: {name} is saved as {saved.args[name]!r} "
                                     f"but the equations give {node.args[name]!r}")
            elif name in saved.args:
                arg = spec.args.get(name)
                if arg is not None and arg.omitted(saved.args[name]):
                    benign.append(f"node {saved.id}: dropped {name}={saved.args[name]!r} (default of an added argument)")
                else:
                    conflicts.append(f"node {saved.id}: {name}={saved.args[name]!r} is saved but the "
                                     "equations drop it")
            else:
                origin = node.provenance.get(name)
                value = node.args[name]
                arg = spec.args.get(name)
                if origin == "operator default":
                    benign.append(f"node {saved.id}: filled {name}={value!r} (operator default)")
                elif origin == "inferred" and arg is not None and arg.dim is not None:
                    benign.append(f"node {saved.id}: filled {name}={value!r} (dimension {arg.dim} of a saved port)")
                else:
                    conflicts.append(f"node {saved.id}: {name} is missing and would be chosen again as "
                                     f"{value!r} ({origin}); a saved plan never re-runs that choice")
    return benign, conflicts


def _completed(plan, verified, registry):
    """The saved plan with the arguments re-resolution filled or dropped.

    Saved source and provenance are kept; only the changed arguments take
    their origin from the re-resolution.
    """
    resolved = {node.id: node for node in verified.nodes}
    nodes = []
    for saved in plan.nodes:
        node = resolved[saved.id]
        changed = set(saved.args) ^ set(node.args)
        provenance = {name: saved.provenance.get(name, node.provenance.get(name)) for name in node.args}
        source = saved.source
        if source and "argument_origins" in source:
            origins = {name: origin for name, origin in source["argument_origins"].items() if name not in changed}
            origins.update({name: origin for name, origin in node.source["argument_origins"].items()
                            if name in changed and name in node.args})
            source = {**source, "argument_origins": origins}
        nodes.append(replace(saved, args=node.args, provenance=provenance, source=source))
    return replace(plan, nodes=tuple(nodes), registry=registry)


def validate_concrete_plan(plan, *, registry=None, limits=None):
    """Verify saved concrete equations, never re-run source/callable/policy choice.

    The saved arguments are resolved again as explicit values against the
    saved external contracts. A plan whose re-resolution matches it exactly is
    returned unchanged. One that differs only by values the saved plan already
    determines (see ``_plan_differences``) --- typically a plan written before
    an operator gained a defaulted argument --- is returned completed, with a
    warning naming what was filled and the new semantic digest. Any other
    difference fails with ``E_INTEGRITY`` and lists every mismatch.
    """
    if not isinstance(plan, ResolvedPlan):
        raise HNDLError("E_SCHEMA", "Expected an immutable ResolvedPlan")
    registry = registry or plan.registry or Registry.builtins()
    bounds = _limits(limits)
    if (type(plan.schema_version) is not int or plan.schema_version != 1
            or type(plan.resolution_version) is not int or plan.resolution_version != 1):
        raise HNDLError("E_STATE_VERSION", "Expected plan schema 1 and resolution version 1")
    for node in plan.nodes:
        registry.by_identity(node.op)
        if set(node.input_shapes) != set(node.inputs) or set(node.output_shapes) != set(node.outputs):
            raise HNDLError("E_SCHEMA", "Saved port contracts are incomplete", node=node.id)
        for shape in (*node.input_shapes.values(), *node.output_shapes.values()):
            _contract(shape, f"{node.id} port", bounds, multiples=True)
    graph = Graph(tuple(Node(node.id, node.op, node.args, node.inputs, node.outputs, node.source,
                             initialization=node.initialization, trainability=node.trainability) for node in plan.nodes),
                  plan.input_shape, plan.output_shape, plan.output_ref, plan.dtype, plan.frontend,
                  input_dtype=plan.input_dtype, named_inputs=plan.named_inputs,
                  named_outputs=plan.named_outputs)
    verified = resolve_graph(graph, registry, bounds)
    if plan.semantic_digest == verified.semantic_digest:
        return replace(plan, registry=registry)
    benign, conflicts = _plan_differences(plan, verified, registry)
    if conflicts or not benign:
        details = conflicts or ["the re-resolved plan differs in a field outside node arguments and port shapes"]
        raise HNDLError("E_INTEGRITY", "Saved plan does not satisfy its operators' equations:\n  "
                        + "\n  ".join(details)
                        + "\nRe-resolve it from its source with this release.")
    warnings.warn("Saved plan lacks arguments this release declares; completed it from values the plan "
                  "already determines:\n  " + "\n  ".join(benign)
                  + f"\nIts semantic digest changes from {plan.semantic_digest} to {verified.semantic_digest}; "
                  "save it again to keep the completed form.", stacklevel=2)
    return _completed(plan, verified, registry)
