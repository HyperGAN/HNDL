"""PyTorch construction and execution for concrete HNDL plans."""

from collections import OrderedDict
from contextlib import contextmanager
import copy
from types import MappingProxyType

import torch
from torch import nn

from .errors import HNDLError
from .types import contract_header

# Build metadata a copied network shares with its original: immutable records
# describing the resolved architecture, never the parameters that train.
SHARED_METADATA = frozenset({"plan", "build_receipt", "_port_orders", "_port_dtypes", "_state_names",
                             "_input_names", "_input_dtypes", "_output_dtypes", "_build_dtype"})

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}


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
        self._state_names = tuple(name for name, _ in self.named_parameters()) + tuple(name for name, _ in self.named_buffers())
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

    def _check(self, value, shape, batch, location, dtype):
        dtype = self._effective_dtype(dtype)
        if not isinstance(value, torch.Tensor):
            raise HNDLError("E_RUNTIME", f"{location} must be a tensor")
        expected = tuple(batch if isinstance(d, str) else d for d in shape)
        if tuple(value.shape) != expected or value.ndim == 0 or value.shape[0] <= 0:
            raise HNDLError("E_RUNTIME", f"{location}: expected shape {expected}, got {tuple(value.shape)}")
        if dtype is not None and value.dtype != dtype:
            raise HNDLError("E_RUNTIME", f"{location}: expected dtype {dtype}, got {value.dtype}")
        if value.device != self._runtime_device:
            raise HNDLError("E_RUNTIME", f"{location}: expected device {self._runtime_device}, got {value.device}")

    def _execute(self, inputs):
        leading = inputs[self._input_names[0]]
        batch = leading.shape[0] if isinstance(leading, torch.Tensor) and leading.ndim else None
        values = {}
        for name in self._input_names:
            value = inputs[name]
            self._check(value, self.plan.inputs[name]["shape"], batch, f"input:{name}",
                        self._input_dtypes[name])
            values[f"input:{name}"] = value
        for node in self.plan.nodes:
            bound = []
            dtypes = self._port_dtypes[node.id]
            for port in self._port_orders[node.id]:
                ref = node.inputs[port]
                value = values[ref]
                self._check(value, node.input_shapes[port], batch, f"{node.id}/{port}", dtypes[port])
                bound.append(value)
            result = self.nodes[f"n_{node.id}"](*bound)
            if len(node.outputs) == 1:
                results = (result,)
            elif isinstance(result, dict) and set(result) == set(node.outputs):
                results = tuple(result[port] for port in node.outputs)
            elif isinstance(result, (tuple, list)) and len(result) == len(node.outputs):
                results = tuple(result)
            else:
                raise HNDLError("E_RUNTIME", f"{node.id}: expected output ports {node.outputs}")
            for port, value in zip(node.outputs, results):
                self._check(value, node.output_shapes[port], batch, f"{node.id}/{port}", dtypes[port])
                values[f"node:{node.id}/{port}"] = value
        outputs = {}
        for name, entry in self.plan.outputs.items():
            value = values[entry["ref"]]
            self._check(value, entry["shape"], batch, name, self._output_dtypes[name])
            outputs[name] = value
        state_names = tuple(name for name, _ in self.named_parameters()) + tuple(name for name, _ in self.named_buffers())
        if state_names != self._state_names:
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
    kwargs = dict(node.args)
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
            # float.hex distinguishes +0.0 and -0.0, which fill_ preserves.
            previous = destination.get(identity)
            if identity in destination and (
                previous.hex() != value.hex() if code == "E_INITIALIZATION" else previous != value
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
        if constant is not None and parameter.dtype != dtype:
            raise HNDLError("E_INITIALIZATION", f"Parameter {names[identity][0]!r} has unsupported constant-initialization dtype {parameter.dtype}; expected {dtype}", node=node.id)
        if flag is True and not (parameter.is_floating_point() or parameter.is_complex()):
            raise HNDLError("E_TRAINABILITY", f"Parameter {names[identity][0]!r} with dtype {parameter.dtype} cannot require gradients", node=node.id)
        result.append((parameter, constant, flag))
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
            for assignments in settings:
                for parameter, constant, trainable in assignments:
                    if constant is not None:
                        parameter.fill_(constant)
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
