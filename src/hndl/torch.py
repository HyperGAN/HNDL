"""PyTorch construction and execution for concrete HNDL plans."""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
import copy
from types import MappingProxyType

import torch
from torch import nn

from .errors import HNDLError
from .settings import MATRIX_SCHEMES
from .types import batch_multiple, contract_header

# Build metadata a copied network shares with its original: immutable records
# describing the resolved architecture, never the parameters that train.
SHARED_METADATA = frozenset({"plan", "build_receipt", "_port_orders", "_port_dtypes",
                             "_input_names", "_input_dtypes", "_output_dtypes", "_build_dtype",
                             "_spec_inputs", "_spec_outputs", "_spec_in", "_spec_out"})

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}

#: Distinguishes "no batch resolved yet" from a legitimately resolved ``None``.
_UNSET = object()

#: The resolved-shape cache before any call: batch, dtype, then three programs.
_UNCOMPILED = (_UNSET, None, (), (), ())


def _compile_shape(shape):
    """A contract shape with its batch entries pre-parsed, once, at build time.

    Every dimension becomes either a plain ``int`` or a
    ``(multiple_or_None, original_text)`` pair, so resolving a port against a
    concrete batch is a multiplication rather than a regular-expression match.
    """
    return tuple(d if not isinstance(d, str) else (batch_multiple(d), d) for d in shape)


class _NodeModules(nn.ModuleDict):
    """Registered modules whose graph structure is fixed after construction."""

    def __init__(self, modules):
        self._locked = False
        super().__init__(modules)
        self._locked = True

    def add_module(self, name, module):
        if self._locked:
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().add_module(name, module)

    def __delitem__(self, key):
        raise TypeError("HNDL architecture is fixed; resolve and build a new plan")

    def clear(self):
        raise TypeError("HNDL architecture is fixed; resolve and build a new plan")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name in self._modules:
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._modules:
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__delattr__(name)


def _shape_text(shape):
    return "[" + ", ".join(map(str, shape)) + "]"


def _is_chain(plan):
    """A chain has one external input feeding one unary node after another."""
    if len(plan.inputs) != 1 or len(plan.outputs) != 1:
        return False
    previous = f"input:{next(iter(plan.inputs))}"
    for node in plan.nodes:
        if (len(node.inputs) != 1 or tuple(node.inputs.values()) != (previous,)
                or len(node.outputs) != 1):
            return False
        previous = f"node:{node.id}/{node.outputs[0]}"
    return plan.output_ref == previous


class GraphModule(nn.Module):
    """Execute a frozen graph once per node and return named output tensors."""

    def __init__(self, plan, modules, device, receipt, port_orders):
        super().__init__()
        self.plan = plan
        self.nodes = _NodeModules(modules)
        self._runtime_device = device
        # The compute dtype the plan was built at, and the one it runs at now:
        # casting the module with .double(), .half() or .to(dtype=...) moves the
        # second without rewriting the build-time declarations below.
        self._build_dtype = DTYPES[plan.dtype]
        self._runtime_dtype = self._build_dtype
        self._chain = _is_chain(plan)
        self._port_orders = port_orders
        self._input_names = tuple(plan.inputs)
        self._input_dtypes = MappingProxyType(
            {name: DTYPES[entry["dtype"]] for name, entry in plan.inputs.items()})
        self._port_dtypes = {}
        produced = {f"input:{name}": entry["dtype"] for name, entry in plan.inputs.items()}
        for node in plan.nodes:
            spec = plan.registry.by_identity(node.op)
            declared = {port: spec.port_dtype(port, plan.dtype) for port in (*node.inputs, *node.outputs)}
            self._port_dtypes[node.id] = {port: None if name == "any" else DTYPES[name] for port, name in declared.items()}
            for port in node.outputs:
                produced[f"node:{node.id}/{port}"] = plan.dtype if declared[port] == "any" else declared[port]
        self._output_dtypes = MappingProxyType(
            {name: DTYPES[produced[entry["ref"]]] for name, entry in plan.outputs.items()})
        self._state_program = self._state_snapshot()
        # Contract shapes with their batch entries pre-parsed; _resolve_shapes
        # turns these into concrete expectations once per distinct batch size.
        self._spec_inputs = {name: _compile_shape(entry["shape"]) for name, entry in plan.inputs.items()}
        self._spec_outputs = {name: _compile_shape(entry["shape"]) for name, entry in plan.outputs.items()}
        self._spec_in = {node.id: {port: _compile_shape(shape) for port, shape in node.input_shapes.items()}
                         for node in plan.nodes}
        self._spec_out = {node.id: {port: _compile_shape(shape) for port, shape in node.output_shapes.items()}
                          for node in plan.nodes}
        self._compiled = _UNCOMPILED
        self.build_receipt = MappingProxyType(receipt)

    def __setattr__(self, name, value):
        if name in ("nodes", "plan") and hasattr(self, name):
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ("nodes", "plan"):
            raise TypeError("HNDL architecture is fixed; resolve and build a new plan")
        super().__delattr__(name)

    def __deepcopy__(self, memo):
        """Copy every module, parameter and buffer; share the resolved plan.

        ``nn.Module`` has no copy protocol of its own, so the default one
        pickles the instance dictionary, which the plan's frozen mappings
        refuse. Copying the dictionary entry by entry instead preserves the
        module internals exactly --- parameters with their ``requires_grad``
        flags, buffers, submodules, hooks and the training mode --- while the
        architecture records are shared rather than duplicated.
        """
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            # The architecture lock guards attribute writes; this rebuilds the
            # instance dictionary directly, exactly as unpickling would.
            object.__setattr__(result, name, value if name in SHARED_METADATA
                               else copy.deepcopy(value, memo))
        # Cheap insurance: the clone rebuilds its baked program against its own
        # modules rather than trusting a structure copied mid-flight.
        object.__setattr__(result, "_compiled", _UNCOMPILED)
        return result

    def __copy__(self):
        """A second handle on the same graph: every tensor stays shared."""
        result = type(self).__new__(type(self))
        result.__dict__.update(self.__dict__)
        return result

    def __reduce__(self):
        raise TypeError(
            f"{type(self).__name__} cannot be pickled because its plan holds frozen mappings; "
            "save plan.to_json() next to torch.save(model.state_dict()) and rebuild with "
            "hndl.torch.build(), or use copy.deepcopy() for an in-memory copy")

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # An empty tensor also tracks device moves and dtype casts for
        # parameterless graphs, which hold no state of their own to follow.
        probe = fn(torch.empty(0, device=self._runtime_device, dtype=self._runtime_dtype))
        self._runtime_device = probe.device
        # ``.double()``, ``.half()`` and ``.to(dtype=...)`` cast floating state
        # only; a probe that came back integral means no compute-dtype change.
        if probe.is_floating_point() or probe.is_complex():
            self._runtime_dtype = probe.dtype
        return self

    def _effective_dtype(self, dtype):
        """The dtype a port carries now, after any floating-point cast.

        Ports declared at the plan's compute dtype follow the module through
        ``.double()``, ``.half()`` and ``.to(dtype=...)``, exactly as its
        parameters do. Integer and boolean ports --- token ids, masks --- and
        ports declared ``any`` are untouched, because those casts leave
        non-floating state alone.
        """
        return self._runtime_dtype if dtype is not None and dtype == self._build_dtype else dtype

    def _state_snapshot(self):
        """Record what ``named_parameters``/``named_buffers`` see, per module.

        The dotted names those walks produce are a pure function of three
        things: the order ``modules()`` visits, each module's path from the
        root, and the keys each module contributes. Freezing the visited
        modules in one flat tuple --- alongside the child mapping that fixes
        every path --- lets :meth:`_execute` re-derive the same answer without
        recursing through ``named_modules`` or building a single string.

        Each row is ``(module, parameter_keys, buffer_keys, children)``.
        ``children`` is the module's ``_modules`` mapping as key/value pairs,
        so a submodule swapped out under an unchanged key is still a change.
        The key tuples are filtered exactly as PyTorch filters them: ``None``
        slots are skipped, and a tensor already seen earlier in the walk is
        skipped, with parameters and buffers de-duplicated independently.
        """
        program, seen_parameters, seen_buffers = [], set(), set()
        for module in self.modules():
            keys = []
            for store, seen in ((module._parameters, seen_parameters),
                                (module._buffers, seen_buffers)):
                contributed = []
                for key, value in store.items():
                    if value is None or id(value) in seen:
                        continue
                    seen.add(id(value))
                    contributed.append(key)
                keys.append(tuple(contributed))
            program.append((module, keys[0], keys[1], tuple(module._modules.items())))
        return tuple(program)

    @staticmethod
    def _resolve(spec, batch):
        """A compiled shape as concrete sizes, or the entry that is not a batch axis.

        A port a batch-axis join or split produced carries ``"k*B"``, which
        means ``k`` times this call's batch, not the batch itself. An entry that
        is symbolic but not a batch dimension comes back as the original string,
        which :meth:`_check` reports from the port that actually carries it ---
        keeping that a runtime error rather than promoting it to build time.
        """
        resolved = []
        for dimension in spec:
            if type(dimension) is tuple:
                multiple, text = dimension
                if multiple is None:
                    return text
                resolved.append(None if batch is None else multiple * batch)
            else:
                resolved.append(dimension)
        return tuple(resolved)

    def _resolve_shapes(self, batch):
        """Bind every port's expected shape and dtype to this batch, once.

        The whole program is published as a single tuple, so a caller on
        another thread reads either the entire previous program or the entire
        new one --- never a half-rebuilt mixture of the two.
        """
        resolve, effective = self._resolve, self._effective_dtype
        expected_in = {node: {p: resolve(s, batch) for p, s in ports.items()}
                       for node, ports in self._spec_in.items()}
        expected_out = {node: {p: resolve(s, batch) for p, s in ports.items()}
                        for node, ports in self._spec_out.items()}
        effdt = {node: {p: effective(d) for p, d in ports.items()}
                 for node, ports in self._port_dtypes.items()}
        compiled = (
            batch, self._runtime_dtype,
            tuple((name, f"input:{name}", resolve(self._spec_inputs[name], batch),
                   effective(self._input_dtypes[name]))
                  for name in self._input_names),
            self._build_program(expected_in, expected_out, effdt),
            tuple((name, entry["ref"], resolve(self._spec_outputs[name], batch),
                   effective(self._output_dtypes[name]))
                  for name, entry in self.plan.outputs.items()))
        self._compiled = compiled
        return compiled

    def _build_program(self, expected_in, expected_out, effdt):
        """Bake the per-call constants --- labels, module handles, port sets --- into tuples."""
        return tuple(
            (self.nodes[f"n_{node.id}"],
             tuple((node.inputs[port], expected_in[node.id][port], f"{node.id}/{port}",
                    effdt[node.id][port])
                   for port in self._port_orders[node.id]),
             tuple((f"node:{node.id}/{port}", expected_out[node.id][port], f"{node.id}/{port}",
                    effdt[node.id][port])
                   for port in node.outputs),
             node.outputs, frozenset(node.outputs), node.id)
            for node in self.plan.nodes)

    def _check(self, value, expected, location, dtype):
        if not isinstance(value, torch.Tensor):
            raise HNDLError("E_RUNTIME", f"{location} must be a tensor")
        if type(expected) is str:
            raise HNDLError("E_RUNTIME", f"{location}: contract entry {expected!r} is not a batch dimension")
        if value.shape != expected or value.ndim == 0 or value.shape[0] <= 0:
            raise HNDLError("E_RUNTIME", f"{location}: expected shape {expected}, got {tuple(value.shape)}")
        if dtype is not None and value.dtype != dtype:
            raise HNDLError("E_RUNTIME", f"{location}: expected dtype {dtype}, got {value.dtype}")
        if value.device != self._runtime_device:
            raise HNDLError("E_RUNTIME", f"{location}: expected device {self._runtime_device}, got {value.device}")

    def _execute(self, inputs):
        leading = inputs[self._input_names[0]]
        batch = leading.shape[0] if isinstance(leading, torch.Tensor) and leading.ndim else None
        # The resolved expectations depend on the batch and on the dtype casts
        # ``_apply`` records; the device is read live by ``_check`` instead, so
        # ``.to(device)`` needs no invalidation.
        compiled = self._compiled
        if batch != compiled[0] or self._runtime_dtype is not compiled[1]:
            compiled = self._resolve_shapes(batch)
        _, _, input_program, node_program, output_program = compiled
        check = self._check
        values = {}
        for name, key, expected, dtype in input_program:
            value = inputs[name]
            check(value, expected, key, dtype)
            values[key] = value
        for module, ins, outs, out_ports, out_set, node_id in node_program:
            bound = []
            for ref, expected, location, dtype in ins:
                value = values[ref]
                check(value, expected, location, dtype)
                bound.append(value)
            result = module(*bound)
            if isinstance(result, dict) and set(result) == out_set:
                results = tuple(result[port] for port in out_ports)
            elif isinstance(result, (tuple, list)) and len(result) == len(out_ports):
                results = tuple(result)
            elif len(out_ports) == 1:
                results = (result,)
            else:
                raise HNDLError("E_RUNTIME", f"{node_id}: expected output ports {out_ports}")
            for (key, expected, location, dtype), value in zip(outs, results):
                check(value, expected, location, dtype)
                values[key] = value
        outputs = {}
        for name, ref, expected, dtype in output_program:
            value = values[ref]
            check(value, expected, name, dtype)
            outputs[name] = value
        # Replay the build-time walk against the flat program instead of
        # recursing through the module tree again. Pinning every module's child
        # mapping keeps the two trees identical --- nothing can be grafted in
        # unseen --- so comparing each module's contributed keys answers exactly
        # what comparing the dotted name tuples used to answer. ``set.add``
        # returns None, so its clause always passes and only records the tensor.
        seen_parameters, seen_buffers = set(), set()
        for module, parameters, buffers, children in self._state_program:
            if (tuple(module._modules.items()) != children
                    or tuple([key for key, value in module._parameters.items()
                              if value is not None and id(value) not in seen_parameters
                              and not seen_parameters.add(id(value))]) != parameters
                    or tuple([key for key, value in module._buffers.items()
                              if value is not None and id(value) not in seen_buffers
                              and not seen_buffers.add(id(value))]) != buffers):
                raise HNDLError("E_RUNTIME", "A module created or removed registered state during forward")
        return outputs

    def _bind(self, args, kwargs):
        """Bind runtime tensors to the declared external inputs, in order."""
        names = self._input_names
        expected = "Expected exactly the external tensor inputs " + ", ".join(repr(n) for n in names)
        if len(args) > len(names):
            raise HNDLError("E_BINDING", expected)
        bound = dict(zip(names, args))
        for name, value in kwargs.items():
            if name in bound:
                raise HNDLError("E_BINDING", f"Input {name!r} was supplied twice")
            bound[name] = value
        if set(bound) != set(names):
            raise HNDLError("E_BINDING", expected)
        return bound

    def forward(self, **inputs):
        return self._execute(self._bind((), inputs))

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self.nodes[f"n_{key}"]
            except KeyError:
                raise KeyError(key) from None
        if not self._chain:
            raise TypeError("Branched graphs support node-name lookup only")
        layers = tuple(self.nodes.values())
        if isinstance(key, slice):
            selected = tuple(self.plan.nodes)[key]
            return nn.Sequential(OrderedDict((f"n_{n.id}", self.nodes[f"n_{n.id}"]) for n in selected))
        if type(key) is int:
            return layers[key]
        raise TypeError("Layer index must be an integer, slice, or node name")

    def __len__(self):
        if not self._chain:
            raise TypeError("Branched graphs support node-name lookup only")
        return len(self.plan.nodes)

    def __iter__(self):
        if not self._chain:
            raise TypeError("Branched graphs support node-name lookup only")
        return iter(self.nodes.values())

    def _alias(self, op):
        registry = self.plan.registry
        try:
            return registry.by_identity(op).alias if registry is not None else op.split("@")[0]
        except HNDLError:
            return op.split("@")[0]

    def __repr__(self):
        lines = [contract_header(self.plan, type(self).__name__)]
        rows = [("index", "name", "operation", "input shape", "output shape")
                if self._chain else ("name", "operation", "input shapes", "output shapes")]
        for i, node in enumerate(self.plan.nodes):
            operation = self._alias(node.op)
            if self._chain:
                rows.append((str(i), node.id, operation,
                             _shape_text(next(iter(node.input_shapes.values()))),
                             _shape_text(next(iter(node.output_shapes.values())))))
            else:
                inputs = ", ".join(f"{p}={node.inputs[p]}:{_shape_text(s)}" for p, s in node.input_shapes.items())
                outputs = ", ".join(f"{p}={_shape_text(s)}" for p, s in node.output_shapes.items())
                rows.append((node.id, operation, inputs, outputs))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
                     for row in rows)
        return "\n".join(lines)


class Network(GraphModule):
    """Tensor facade over the graph: one tensor out, or a dict of named outputs."""

    def forward(self, *inputs, **named):
        results = self._execute(self._bind(inputs, named))
        if len(results) == 1:
            return next(iter(results.values()))
        return results


@contextmanager
def _initialization_rng(device, seed):
    if seed is None:
        yield
        return
    if type(seed) is not int:
        raise HNDLError("E_SCHEMA", "initialization_seed must be an integer or None")
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # torch.manual_seed seeds every CUDA device; only touch this scope.
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        yield


def _symbol_values(spec, node):
    """Resolved values of the shape symbols a module constructor requests."""
    values = {}
    shapes = {**node.input_shapes, **node.output_shapes}
    from .operator import ELLIPSIS, Sym
    for port in (*spec.inputs, *spec.outputs):
        if port.pattern is None or port.name not in shapes:
            continue
        shape = shapes[port.name]
        pattern = port.pattern
        if ELLIPSIS in pattern:
            index = pattern.index(ELLIPSIS)
            middle = len(shape) - (len(pattern) - 1)
            pattern = pattern[:index] + (None,) * middle + pattern[index + 1:]
        for axis, dimension in enumerate(pattern):
            if isinstance(dimension, Sym) and dimension.name not in values:
                values[dimension.name] = shape[axis] // dimension.scale
    return values


def construct(spec, node, device, dtype):
    """Instantiate one operator's module from concrete plan arguments."""
    # A resolved node omits arguments added after release that hold their
    # default (``Arg(since=...)``); the module still receives every argument.
    kwargs = {name: arg.default for name, arg in spec.args.items() if arg.since is not None}
    kwargs.update(node.args)
    if spec.init_symbols:
        symbols = _symbol_values(spec, node)
        for name in spec.init_symbols:
            kwargs[name] = symbols[name]
    for name in spec.init_shapes:
        kwargs[name] = {port: tuple(shape) for port, shape in getattr(node, name).items()}
    with torch.device(device):
        layer = spec.module(**kwargs)
    if not isinstance(layer, nn.Module):
        raise HNDLError("E_REGISTRY", f"{node.id}: operator {spec.alias} did not construct an nn.Module")
    if torch.device(device).type == "meta":
        # The allocation-free probe only measures; never move a module onto meta.
        return layer
    return layer.to(device=device, dtype=dtype)


def _state_bytes(module, compute_dtype=None):
    """Unique registered parameter/buffer storage, including nonpersistent buffers.

    With ``compute_dtype`` the count assumes floating tensors will be cast to
    it, which is how the allocation-free probe predicts the real footprint.
    """
    storages = {}
    for value in (*module.parameters(), *module.buffers()):
        if compute_dtype is not None:
            size = torch.finfo(compute_dtype).bits // 8 if value.is_floating_point() else value.element_size()
            storages[id(value)] = value.numel() * size
            continue
        if value.device.type == "meta":
            storages[id(value)] = value.numel() * value.element_size()
            continue
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr())
        storages[key] = max(storages.get(key, 0), storage.nbytes())
    return sum(storages.values())


def parameter_counts(plan, *, registry=None):
    """Per-node parameter counts from an allocation-free meta-device construction."""
    from .registry import Registry
    registry = registry if registry is not None else plan.registry or Registry.builtins()
    counts = {}
    for node in plan.nodes:
        spec = registry.by_identity(node.op)
        try:
            with registry.activated():
                layer = construct(spec, node, "meta", DTYPES[plan.dtype])
        except Exception:
            counts[node.id] = None
            continue
        counts[node.id] = sum(p.numel() for p in layer.parameters())
    return counts


INITIALIZERS = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "truncated_normal": nn.init.trunc_normal_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
    "orthogonal": nn.init.orthogonal_,
}


def _initializer_identity(value):
    """Compare two overrides exactly, distinguishing +0.0 from -0.0."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, item.hex() if type(item) is float else item)
                            for key, item in value.items()))
    # float.hex distinguishes +0.0 and -0.0, which fill_ preserves.
    return value.hex()


def _apply_initializer(parameter, override, name, node_id):
    """Fill one parameter in place; scheme draws come from the active RNG."""
    if not isinstance(override, Mapping):
        parameter.fill_(override)
        return
    kind = override["kind"]
    arguments = {key: item for key, item in override.items() if key != "kind"}
    try:
        INITIALIZERS[kind](parameter, **arguments)
    except (ValueError, RuntimeError) as exc:
        raise HNDLError("E_INITIALIZATION",
                        f"Initializer {kind} cannot initialize parameter {name!r} with shape "
                        f"{tuple(parameter.shape)}: {exc}", node=node_id) from None


def _prepare_parameter_settings(layer, node, dtype):
    """Validate exact targets and aliases without changing any parameter."""
    parameters = dict(layer.named_parameters(remove_duplicate=False))
    buffers = dict(layer.named_buffers(remove_duplicate=False))
    initialization = node.initialization["overrides"]
    trainability = node.trainability["overrides"]
    default_trainability = node.trainability["default"]

    for code, assignments in (("E_INITIALIZATION", initialization), ("E_TRAINABILITY", trainability)):
        for name in sorted(assignments):
            if name not in parameters:
                detail = "is a buffer, not a parameter" if name in buffers else "is not a registered parameter"
                raise HNDLError(code, f"Parameter target {name!r} {detail}", node=node.id)

    by_id, names, storage_groups, buffer_storages = {}, {}, {}, set()
    for name, parameter in parameters.items():
        identity = id(parameter)
        by_id[identity] = parameter
        names.setdefault(identity, []).append(name)
        storage = parameter.untyped_storage()
        if storage.nbytes():
            key = (parameter.device, storage.data_ptr())
            storage_groups.setdefault(key, set()).add(identity)
    for buffer in buffers.values():
        storage = buffer.untyped_storage()
        if storage.nbytes():
            buffer_storages.add((buffer.device, storage.data_ptr()))

    constants, flags = {}, {}
    for assignments, destination, code in (
        (initialization, constants, "E_INITIALIZATION"),
        (trainability, flags, "E_TRAINABILITY"),
    ):
        for name in sorted(assignments):
            identity = id(parameters[name])
            value = assignments[name]
            previous = destination.get(identity)
            if identity in destination and (
                _initializer_identity(previous) != _initializer_identity(value)
                if code == "E_INITIALIZATION" else previous != value
            ):
                raise HNDLError(code, f"Conflicting settings for aliases of parameter {names[identity]!r}", node=node.id)
            destination[identity] = value

    effective_flags = {identity: flags.get(identity, parameter.requires_grad if default_trainability is None
                                           else default_trainability)
                       for identity, parameter in by_id.items()}
    for key, identities in storage_groups.items():
        if identities & constants.keys():
            if len(identities) > 1:
                raise HNDLError("E_INITIALIZATION", "Constant override targets distinct parameters sharing storage; use one Parameter with named aliases instead", node=node.id)
            if key in buffer_storages:
                raise HNDLError("E_INITIALIZATION", "Constant override targets parameter storage also registered as a buffer", node=node.id)
        touched = default_trainability is not None or bool(identities & flags.keys())
        if touched and len({effective_flags[identity] for identity in identities}) > 1:
            raise HNDLError("E_TRAINABILITY", "Parameters sharing storage have conflicting effective trainability settings", node=node.id)

    result = []
    for identity, parameter in by_id.items():
        constant = constants.get(identity)
        flag = effective_flags[identity] if default_trainability is not None or identity in flags else None
        target = names[identity][0]
        if constant is not None and parameter.dtype != dtype:
            kind = "constant" if not isinstance(constant, Mapping) else f"{constant['kind']}"
            raise HNDLError("E_INITIALIZATION", f"Parameter {target!r} has unsupported {kind}-initialization dtype {parameter.dtype}; expected {dtype}", node=node.id)
        # Rank is checked before any node is touched so a rejected scheme never
        # leaves an earlier node's parameters half initialized.
        if isinstance(constant, Mapping) and constant["kind"] in MATRIX_SCHEMES and parameter.dim() < 2:
            raise HNDLError("E_INITIALIZATION", f"Initializer {constant['kind']} requires a parameter of at least two dimensions; {target!r} has shape {tuple(parameter.shape)}", node=node.id)
        if flag is True and not (parameter.is_floating_point() or parameter.is_complex()):
            raise HNDLError("E_TRAINABILITY", f"Parameter {target!r} with dtype {parameter.dtype} cannot require gradients", node=node.id)
        result.append((parameter, target, constant, flag))
    return result


def _build(plan, *, device, initialization_seed=None, registry=None, facade=False, limits=None):
    if plan.dtype not in ("float32", "float16", "bfloat16"):
        raise HNDLError("E_SCHEMA", f"Unsupported plan dtype {plan.dtype!r}")
    dtype = DTYPES[plan.dtype]
    device = torch.device(device)
    if device.type == "cpu":
        device = torch.device("cpu")
    if device.type not in ("cpu", "cuda"):
        raise HNDLError("E_SCHEMA", f"Unsupported execution device {device}; use cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise HNDLError("E_RUNTIME", "CUDA was requested but is unavailable")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if device.index >= torch.cuda.device_count():
            raise HNDLError("E_RUNTIME", f"CUDA device {device.index} is unavailable")
    registry = registry if registry is not None else plan.registry
    if registry is None:
        from .registry import Registry
        registry = Registry.builtins()
    from .resolver import validate_concrete_plan, _limits
    bounds = _limits(limits)
    plan = validate_concrete_plan(plan, registry=registry, limits=limits)
    specs = {node.id: registry.by_identity(node.op) for node in plan.nodes}
    port_orders = {node.id: specs[node.id].input_ports_for(node.args) for node in plan.nodes}

    # Bound registered storage with an allocation-free meta construction first.
    total = 0
    unbounded = []
    for node in plan.nodes:
        try:
            with registry.activated():
                probe = construct(specs[node.id], node, "meta", dtype)
        except Exception:
            unbounded.append(node.id)
            continue
        total += _state_bytes(probe, dtype)
        if total > bounds["max_state_bytes"]:
            raise HNDLError("E_RESOURCE", "Plan exceeds max_state_bytes before allocation", node=node.id)

    modules = OrderedDict()
    module_owners, tensor_owners, storage_owners = {}, {}, {}

    def check_ownership(layer, node_id):
        owned = [(module_owners, id(child)) for child in layer.modules()]
        for tensor in (*layer.parameters(), *layer.buffers()):
            owned.append((tensor_owners, id(tensor)))
            storage = tensor.untyped_storage()
            if storage.nbytes():
                owned.append((storage_owners, (tensor.device, storage.data_ptr())))
        for owners, key in owned:
            if key in owners and owners[key] != node_id:
                raise HNDLError("E_REGISTRY", f"{node_id}: operator reuses a module or registered state from node {owners[key]}; each node must own an independent instance")
        for owners, key in owned:
            owners[key] = node_id

    with registry.activated(), _initialization_rng(device, initialization_seed):
        for node in plan.nodes:
            layer = construct(specs[node.id], node, device, dtype)
            check_ownership(layer, node.id)
            if node.id in unbounded:
                total += _state_bytes(layer)
                if total > bounds["max_state_bytes"]:
                    raise HNDLError("E_RESOURCE", "Plan exceeds max_state_bytes", node=node.id)
            modules[f"n_{node.id}"] = layer
        # Resolve every target before applying settings to any graph node.
        settings = [_prepare_parameter_settings(modules[f"n_{node.id}"], node, dtype) for node in plan.nodes]
        with torch.no_grad():
            for node, assignments in zip(plan.nodes, settings):
                for parameter, target, override, trainable in assignments:
                    if override is not None:
                        _apply_initializer(parameter, override, target, node.id)
                    if trainable is not None:
                        parameter.requires_grad_(trainable)
    receipt = {"torch_version": str(torch.__version__), "device": str(device), "dtype": plan.dtype,
               "initialization_seed": initialization_seed,
               "seed_mode": "caller" if initialization_seed is None else "isolated",
               "state_bytes": total}
    cls = Network if facade else GraphModule
    return cls(plan, modules, device, receipt, port_orders)


def build(plan, *, device, initialization_seed=None, registry=None, limits=None):
    """Build a concrete plan; forward accepts named inputs and returns a dict."""
    return _build(plan, device=device, initialization_seed=initialization_seed, registry=registry, limits=limits)


def _network(resolve_name, source, *, input_shape, output_shape, device,
             dtype="float32", registry=None, initialization_seed=None, limits=None, input_dtype=None):
    import hndl
    resolve = getattr(hndl, resolve_name)
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape,
                   dtype=dtype, registry=registry, limits=limits, input_dtype=input_dtype)
    return _build(plan, device=device, initialization_seed=initialization_seed,
                  registry=registry, facade=True, limits=limits)


def network(source, *, input_shape, output_shape, device, dtype="float32",
            registry=None, initialization_seed=None, limits=None, input_dtype=None):
    """Resolve declarative source and return a tensor-in/tensor-out module."""
    return _network("resolve", source, input_shape=input_shape, output_shape=output_shape,
                    device=device, dtype=dtype, registry=registry,
                    initialization_seed=initialization_seed, limits=limits, input_dtype=input_dtype)


def network_file(path, *, input_shape, output_shape, device, dtype="float32",
                 registry=None, initialization_seed=None, limits=None, input_dtype=None):
    """Read declarative UTF-8 source and construct its resolved network."""
    return _network("resolve_file", path, input_shape=input_shape, output_shape=output_shape,
                    device=device, dtype=dtype, registry=registry,
                    initialization_seed=initialization_seed, limits=limits, input_dtype=input_dtype)


def network_from_callable(fn, *, input_shape, output_shape, device, dtype="float32",
                          registry=None, initialization_seed=None, limits=None, input_dtype=None):
    """Capture trusted Python exactly once, then construct the resolved network."""
    return _network("resolve_callable", fn, input_shape=input_shape, output_shape=output_shape,
                    device=device, dtype=dtype, registry=registry,
                    initialization_seed=initialization_seed, limits=limits, input_dtype=input_dtype)
