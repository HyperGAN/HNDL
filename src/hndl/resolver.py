"""Bounded, monotone tensor-shape resolution without importing torch."""

from dataclasses import replace
from collections.abc import Mapping
from math import prod
import re

from .errors import HNDLError
from .registry import Registry, normalize_arguments, preserves_shape
from .schema import Dim, ShapeRule
from .types import Graph, Node, ResolvedNode, ResolvedPlan


DEFAULT_LIMITS = {
    "max_nodes": 4096, "max_edges": 16_384, "max_dimension": 1_048_576,
    "max_elements": 268_435_456, "max_state_bytes": 1_073_741_824,
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


def _contract(shape, label, limits):
    if not isinstance(shape, (tuple, list)) or len(shape) not in (2, 4):
        raise HNDLError("E_SCHEMA", f"{label} must include batch and have rank 2 (BF) or 4 (NCHW)")
    for axis, value in enumerate(shape):
        if axis == 0 and value == "B":
            continue
        if type(value) is not int or value <= 0:
            raise HNDLError("E_SCHEMA", f"{label}[{axis}] must be positive, with only batch allowed to be 'B'")
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
        if tuple(node.outputs) != spec.output_ports:
            raise HNDLError("E_BINDING", f"{node.id} output ports must be {spec.output_ports}", node=node.id)
        if node.source is not None and not isinstance(node.source, Mapping):
            raise HNDLError("E_SCHEMA", "Node source metadata must be a mapping", node=node.id)
        args = normalize_arguments(spec, (), node.args)
        if spec.variadic_inputs and args["input_count"] > limits["max_edges"]:
            raise HNDLError("E_RESOURCE", "concat input_count exceeds max_edges", node=node.id)
        source = dict(node.source or {})
        origins = {key: "explicit" if key in node.args else "operator default" for key in args}
        if "policy" in node.args:
            source["policy"] = "spatial.up2_transpose@1"
            for key in ("kernel_size", "stride", "padding", "dilation", "output_padding", "groups"):
                if key not in node.args:
                    origins[key] = "policy-selected"
        origins.update(source.get("argument_origins", {}))
        source["argument_origins"] = origins
        ports = tuple(f"x{i}" for i in range(args["input_count"])) if spec.variadic_inputs else spec.input_ports
        if set(node.inputs) != set(ports):
            raise HNDLError("E_BINDING", f"Required input ports are {ports}, got {tuple(node.inputs)}", node=node.id)
        lookup[node.id] = replace(node, args=args, source=source)
        specs[node.id] = spec
        declarations[node.id] = index
    refs = {"input:x"}
    refs.update(f"node:{node.id}/{port}" for node in lookup.values() for port in node.outputs)
    if not isinstance(graph.output_ref, str) or graph.output_ref not in refs:
        raise HNDLError("E_BINDING", f"Unknown public output reference {graph.output_ref!r}")
    dependencies = {}
    consumers = {node_id: [] for node_id in lookup}
    for node in lookup.values():
        deps = set()
        for ref in node.inputs.values():
            if not isinstance(ref, str) or ref not in refs:
                raise HNDLError("E_BINDING", f"Unknown input reference {ref!r}", node=node.id)
            if ref != "input:x":
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
    queue = [] if graph.output_ref == "input:x" else [graph.output_ref[5:].split("/", 1)[0]]
    while queue:
        key = queue.pop()
        if key not in reachable:
            reachable.add(key)
            queue.extend(dependencies[key])
    dead = set(lookup) - reachable
    if dead:
        raise HNDLError("E_BINDING", f"Nodes do not reach the selected output: {', '.join(sorted(dead))}")
    return ordered, specs


class _Solver:
    def __init__(self, graph, nodes, specs, limits):
        self.graph, self.nodes, self.specs, self.limits = graph, nodes, specs, limits
        self.batch = graph.input_shape[0]
        self.shapes = {"input:x": list(graph.input_shape)}
        self.args = {node.id: dict(node.args) for node in nodes}
        self.changed = False
        self.node = None
        self.intervals = {}
        self.set_shape(graph.output_ref, graph.output_shape)

    def error(self, code, message):
        source = self.node.source if self.node and self.node.source else {}
        raise HNDLError(code, message, node=self.node.id if self.node else None,
                        line=source.get("line"), column=source.get("column"))

    def set_shape(self, ref, values, code="E_CONSTRAINT"):
        values = list(values)
        if len(values) not in (2, 4):
            self.error(code, f"Unsupported tensor rank {len(values)} at {ref}")
        for axis, value in enumerate(values):
            if value is None:
                continue
            if axis == 0:
                if value != self.batch:
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
                self.error(code, f"Dimension {index} at {ref}: {old[index]} conflicts with required {value}")

    def rank(self, ref, rank):
        self.set_shape(ref, [self.batch] + [None] * (rank - 1))
        return self.shapes[ref]

    def axis(self, ref, axis, value, code="E_CONSTRAINT"):
        shape = self.shapes.get(ref)
        if shape is None or value is None:
            return
        updated = [None] * len(shape)
        updated[0] = self.batch
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

    def apply(self, node):
        self.node = node
        spec = self.specs[node.id]
        kind, args = spec.identity, self.args[node.id]
        ports = node.inputs
        outs = {port: f"node:{node.id}/{port}" for port in node.outputs}
        x = ports.get("x")
        y = outs.get("out")
        if spec.shape is preserves_shape:
            self.equal(ports[spec.input_ports[0]], outs[spec.output_ports[0]])
            return
        if type(spec.shape) is ShapeRule:
            self.patterns(spec.shape, ports, outs)
            return
        if kind in ("relu", "leaky_relu", "tanh", "group_norm"):
            self.equal(x, y)
            if kind == "group_norm" and x in self.shapes:
                channels = self.shapes[x][1]
                if "num_channels" in args:
                    self.axis(x, 1, args["num_channels"])
                    channels = args["num_channels"]
                if channels is not None:
                    if channels % args["num_groups"]:
                        self.error("E_CONSTRAINT", f"num_groups={args['num_groups']} must divide channels={channels}")
                    self.argument(args, "num_channels", channels)
            return
        if kind == "linear":
            a, b = self.rank(x, 2), self.rank(y, 2)
            self.axis(x, 1, args.get("in_features"))
            self.axis(y, 1, args.get("out_features"))
            self.argument(args, "in_features", a[1])
            self.argument(args, "out_features", b[1])
        elif kind in ("reshape", "flatten"):
            if kind == "flatten":
                self.rank(y, 2)
            else:
                prefix = args["shape"]
                if len(prefix) >= 2:
                    self.rank(y, 4)
                if y in self.shapes:
                    if len(prefix) >= len(self.shapes[y]):
                        self.error("E_RESHAPE", "Reshape prefix exceeds the output rank")
                    for axis, value in enumerate(prefix, 1):
                        self.axis(y, axis, value, "E_RESHAPE")
            self.product(x, y)
        elif kind in ("conv2d", "conv_transpose2d"):
            a, b = self.rank(x, 4), self.rank(y, 4)
            self.axis(x, 1, args.get("in_channels"))
            self.axis(y, 1, args.get("out_channels"))
            self.argument(args, "in_channels", a[1])
            self.argument(args, "out_channels", b[1])
            for channels in (a[1], b[1]):
                if channels is not None and channels % args["groups"]:
                    self.error("E_CONSTRAINT", f"groups={args['groups']} must divide input and output channels")
            for j in range(2):
                axis = j + 2
                k, stride, pad, dilation = (args[key][j] for key in ("kernel_size", "stride", "padding", "dilation"))
                i, o = a[axis], b[axis]
                if i is not None:
                    result = ((i + 2 * pad - dilation * (k - 1) - 1) // stride + 1
                              if kind == "conv2d" else
                              (i - 1) * stride - 2 * pad + dilation * (k - 1) + args["output_padding"][j] + 1)
                    self.axis(y, axis, result)
                if o is not None:
                    if kind == "conv_transpose2d":
                        numerator = o + 2 * pad - dilation * (k - 1) - args["output_padding"][j] - 1
                        if numerator % stride:
                            self.error("E_CONSTRAINT", f"Target extent {o} requires a non-integer transpose-convolution input")
                        self.axis(x, axis, numerator // stride + 1)
                    else:
                        lower = max(1, (o - 1) * stride - 2 * pad + dilation * (k - 1) + 1)
                        upper = o * stride - 2 * pad + dilation * (k - 1)
                        if lower > upper:
                            self.error("E_CONSTRAINT", "Convolution inverse has no positive input extent")
                        previous = self.intervals.get((x, axis))
                        if previous:
                            lower, upper = max(lower, previous[0]), min(upper, previous[1])
                        if lower > upper:
                            self.error("E_CONSTRAINT", "Convolution inverse intervals are incompatible")
                        self.intervals[x, axis] = (lower, upper)
                        if i is not None and not lower <= i <= upper:
                            self.error("E_CONSTRAINT", f"Input extent {i} lies outside required [{lower}, {upper}]")
                        if lower == upper:
                            self.axis(x, axis, lower)
        elif kind == "add":
            self.equal(ports["a"], y)
            self.equal(ports["b"], y)
        elif kind == "split":
            first, rest = outs["first"], outs["rest"]
            known = next((self.shapes[r] for r in (x, first, rest) if r in self.shapes), None)
            if known is None:
                return
            dim = args["dim"]
            if dim >= len(known):
                self.error("E_ARGUMENT", f"split dim {dim} is outside rank {len(known)}")
            for ref in (x, first, rest):
                self.rank(ref, len(known))
            for axis in range(1, len(known)):
                if axis != dim:
                    for a_ref in (x, first, rest):
                        for b_ref in (x, first, rest):
                            self.axis(a_ref, axis, self.shapes[b_ref][axis])
            self.axis(first, dim, args.get("size"))
            i, a, b = (self.shapes[ref][dim] for ref in (x, first, rest))
            if a is not None and b is not None:
                self.axis(x, dim, a + b)
            if i is not None and a is not None:
                self.axis(rest, dim, i - a)
            if i is not None and b is not None:
                self.axis(first, dim, i - b)
            self.argument(args, "size", self.shapes[first][dim])
        elif kind == "concat":
            refs = tuple(ports[f"x{i}"] for i in range(args["input_count"]))
            all_refs = refs + (y,)
            known = next((self.shapes[r] for r in all_refs if r in self.shapes), None)
            if known is None:
                return
            axis = args["axis"]
            if axis >= len(known):
                self.error("E_ARGUMENT", f"concat axis {axis} is outside rank {len(known)}")
            for ref in all_refs:
                self.rank(ref, len(known))
            for dim in range(1, len(known)):
                if dim != axis:
                    # All tensors share this extent. Propagate one known value
                    # once per edge; comparing every pair lets a short config
                    # make the parent resolver do quadratic work.
                    extent = next((self.shapes[ref][dim] for ref in all_refs
                                   if self.shapes[ref][dim] is not None), None)
                    for ref in all_refs:
                        self.axis(ref, dim, extent)
            sizes = [self.shapes[ref][axis] for ref in refs]
            missing = [i for i, size in enumerate(sizes) if size is None]
            if not missing:
                self.axis(y, axis, sum(sizes))
            elif len(missing) == 1 and self.shapes[y][axis] is not None:
                self.axis(refs[missing[0]], axis, self.shapes[y][axis] - sum(size for size in sizes if size is not None))
        else:
            self.error("E_UNRESOLVED", f"No pure shape rules for {node.op}")

    def patterns(self, rule, inputs, outputs):
        """Refine shared positive variables in two bounded linear passes.

        Variables live only in this node invocation. Both input and output
        facts participate equally; every shared occurrence is an equation.
        """
        entries = [(inputs[name], pattern) for name, pattern in rule.inputs.items()]
        entries.extend((outputs[name], pattern) for name, pattern in rule.outputs.items())
        values = {}
        for ref, pattern in entries:
            shape = self.rank(ref, len(pattern))
            for axis, dimension in enumerate(pattern):
                if axis == 0:
                    continue  # Pattern validation fixes unscaled B here.
                if type(dimension) is int:
                    self.axis(ref, axis, dimension)
                    continue
                extent = shape[axis]
                if extent is None:
                    continue
                if extent % dimension.scale:
                    self.error("E_CONSTRAINT", f"Extent {extent} at {ref}[{axis}] is not divisible by scale {dimension.scale} of {dimension.name}")
                value = extent // dimension.scale
                if dimension.name in values and values[dimension.name] != value:
                    self.error("E_CONSTRAINT", f"Shared dimension {dimension.name} requires incompatible values {values[dimension.name]} and {value}")
                values[dimension.name] = value
        for ref, pattern in entries:
            for axis, dimension in enumerate(pattern):
                if axis and isinstance(dimension, Dim) and dimension.name in values:
                    self.axis(ref, axis, dimension.scale * values[dimension.name])

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
        state_total = 0
        for node in self.nodes:
            self.node = node
            args = self.args[node.id]
            for port, ref in list(node.inputs.items()) + [(port, f"node:{node.id}/{port}") for port in node.outputs]:
                shape = self.shapes.get(ref)
                if shape is None or any(value is None for value in shape):
                    interval = next((bounds for (r, _), bounds in self.intervals.items() if r == ref), None)
                    detail = f"; inverse allows {interval}" if interval else ""
                    self.error("E_AMBIGUOUS", f"Shape for {port} ({ref}) remains ambiguous{detail}; supply a dimension or a policy")
            if self.specs[node.id].identity == "reshape":
                args["shape"] = tuple(self.shapes[f"node:{node.id}/out"][1:])
            state_bytes = self.state_bound(node, args)
            state_total += state_bytes
            if state_total > self.limits["max_state_bytes"]:
                self.error("E_RESOURCE", "Plan exceeds max_state_bytes before allocation")
            origins = {}
            source_origins = node.source.get("argument_origins", {}) if node.source else {}
            for name in args:
                origins[name] = source_origins.get(name, "explicit" if name in node.args else "inferred")
            resolved.append(ResolvedNode(
                id=node.id, op=node.op, args=args, inputs=node.inputs, outputs=node.outputs, source=node.source,
                input_shapes={key: tuple(self.shapes[ref]) for key, ref in node.inputs.items()},
                output_shapes={key: tuple(self.shapes[f"node:{node.id}/{key}"]) for key in node.outputs},
                state_bytes=state_bytes, state_version=self.specs[node.id].state_version, provenance=origins,
                initialization=node.initialization, trainability=node.trainability,
            ))
        return tuple(resolved)

    def state_bound(self, node, args):
        spec = self.specs[node.id]
        if spec.shape is preserves_shape or type(spec.shape) is ShapeRule:
            return spec.max_state_bytes
        if spec.identity == "linear":
            count = args["in_features"] * args["out_features"] + (args["out_features"] if args["bias"] else 0)
        elif spec.identity in ("conv2d", "conv_transpose2d"):
            count = args["in_channels"] * args["out_channels"] // args["groups"] * prod(args["kernel_size"])
            count += args["out_channels"] if args["bias"] else 0
        elif spec.identity == "group_norm":
            count = 2 * args["num_channels"] if args["affine"] else 0
        else:
            count = 0
        return 4 * count


def resolve_graph(graph, registry=None, limits=None):
    if not isinstance(graph, Graph):
        raise HNDLError("E_SCHEMA", "resolve_graph requires a captured Graph")
    registry = Registry.builtins() if registry is None else registry
    if not isinstance(registry, Registry):
        raise HNDLError("E_REGISTRY", "registry must be an explicit Registry")
    limits = _limits(limits)
    if graph.dtype != "float32":
        raise HNDLError("E_SCHEMA", "Only float32 is qualified in this initial implementation")
    if not isinstance(graph.frontend, str) or len(graph.frontend) > 256:
        raise HNDLError("E_SCHEMA", "Frontend provenance must be a bounded string")
    input_shape = _contract(graph.input_shape, "input_shape", limits)
    output_shape = _contract(graph.output_shape, "output_shape", limits)
    if input_shape[0] != output_shape[0]:
        raise HNDLError("E_CONSTRAINT", "Input and output must declare the same batch dimension")
    nodes, specs = _ordered_nodes(graph, registry, limits)
    solver = _Solver(graph, nodes, specs, limits)
    resolved = solver.run()
    return ResolvedPlan(resolved, input_shape, output_shape, graph.output_ref, graph.dtype, graph.frontend, registry)


def validate_concrete_plan(plan, *, registry=None, limits=None):
    """Verify saved concrete equations, never re-run source/callable/policy choice."""
    if not isinstance(plan, ResolvedPlan):
        raise HNDLError("E_SCHEMA", "Expected an immutable ResolvedPlan")
    registry = registry or plan.registry or Registry.builtins()
    bounds = _limits(limits)
    if (type(plan.schema_version) is not int or plan.schema_version != 1
            or type(plan.resolution_version) is not int or plan.resolution_version != 1):
        raise HNDLError("E_STATE_VERSION", "Expected plan schema 1 and resolution version 1")
    for node in plan.nodes:
        spec = registry.by_identity(node.op)
        if type(node.state_version) is not int or node.state_version != spec.state_version:
            raise HNDLError("E_STATE_VERSION", f"{node.op} requires state version {node.state_version}; registered version is {spec.state_version}", node=node.id)
        if type(node.state_bytes) is not int or node.state_bytes < 0:
            raise HNDLError("E_SCHEMA", "State bounds must be nonnegative integers", node=node.id)
        if set(node.input_shapes) != set(node.inputs) or set(node.output_shapes) != set(node.outputs):
            raise HNDLError("E_SCHEMA", "Saved port contracts are incomplete", node=node.id)
        for shape in (*node.input_shapes.values(), *node.output_shapes.values()):
            _contract(shape, f"{node.id} port", bounds)
    graph = Graph(tuple(Node(node.id, node.op, node.args, node.inputs, node.outputs, node.source,
                             initialization=node.initialization, trainability=node.trainability) for node in plan.nodes),
                  plan.input_shape, plan.output_shape, plan.output_ref, plan.dtype, plan.frontend)
    verified = resolve_graph(graph, registry, bounds)
    if plan.semantic_digest != verified.semantic_digest:
        raise HNDLError("E_INTEGRITY", "Saved concrete arguments/port shapes/state bounds are inconsistent; no inferred replacement is accepted")
    return replace(plan, registry=registry)
