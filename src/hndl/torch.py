"""PyTorch construction and execution for concrete HNDL plans.

Importing this backend is explicit: the resolver itself never imports torch.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from .errors import HNDLError

_BUILTINS = frozenset(f"{name}@1" for name in (
    "linear", "conv2d", "conv_transpose2d", "group_norm", "reshape", "flatten",
    "relu", "leaky_relu", "tanh", "add", "split", "concat",
))


@dataclass(frozen=True)
class TorchBinding:
    module: object
    state_version: int


def register_torch(registry, alias, *, module, state_version):
    """Attach an explicitly supplied constructor to an exact pure operator."""
    entry = registry.get(alias)
    if entry.key in _BUILTINS:
        raise HNDLError("E_REGISTRY", f"Built-in backend {entry.key} cannot be replaced")
    if type(state_version) is not int or state_version != entry.state_version:
        raise HNDLError("E_STATE_VERSION", f"{alias}: backend state version differs from pure registration")
    if entry.key in registry.backends:
        raise HNDLError("E_REGISTRY", f"Backend already registered for {entry.key}")
    if not callable(module):
        raise HNDLError("E_REGISTRY", "module must be a trusted callable constructor")
    registry.backends[entry.key] = TorchBinding(module, state_version)


class _Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def extra_repr(self):
        return f"shape={self.shape}"


class _Add(nn.Module):
    def forward(self, a, b):
        return a + b


class _Split(nn.Module):
    def __init__(self, size, remainder, dim):
        super().__init__()
        self.sizes = (size, remainder)
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.sizes, dim=self.dim)


class _Concat(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, *xs):
        return torch.cat(xs, dim=self.axis)


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
    previous = "input:x"
    for node in plan.nodes:
        if dict(node.inputs) != {"x": previous} or tuple(node.outputs) != ("out",):
            return False
        previous = f"node:{node.id}/out"
    return plan.output_ref == previous


class GraphModule(nn.Module):
    """Execute a frozen graph once per node and return named output tensors."""

    def __init__(self, plan, modules, device, receipt, port_orders):
        super().__init__()
        self.plan = plan
        self.nodes = _NodeModules(modules)
        self._runtime_device = device
        self._chain = _is_chain(plan)
        self._port_orders = port_orders
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

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # An empty tensor also tracks device moves for parameterless graphs.
        probe = fn(torch.empty(0, device=self._runtime_device))
        self._runtime_device = probe.device
        return self

    def _check(self, value, shape, batch, location):
        if not isinstance(value, torch.Tensor):
            raise HNDLError("E_RUNTIME", f"{location} must be a tensor")
        expected = tuple(batch if d == "B" else d for d in shape)
        if tuple(value.shape) != expected or value.ndim == 0 or value.shape[0] <= 0:
            raise HNDLError("E_RUNTIME", f"{location}: expected shape {expected}, got {tuple(value.shape)}")
        if value.dtype != torch.float32:
            raise HNDLError("E_RUNTIME", f"{location}: expected dtype float32, got {value.dtype}")
        if value.device != self._runtime_device:
            raise HNDLError("E_RUNTIME", f"{location}: expected device {self._runtime_device}, got {value.device}")

    def _execute(self, x):
        batch = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim else None
        self._check(x, self.plan.input_shape, batch, "input:x")
        values = {"input:x": x}
        for node in self.plan.nodes:
            bound = []
            for port in self._port_orders[node.id]:
                ref = node.inputs[port]
                value = values[ref]
                self._check(value, node.input_shapes[port], batch, f"{node.id}/{port}")
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
                self._check(value, node.output_shapes[port], batch, f"{node.id}/{port}")
                values[f"node:{node.id}/{port}"] = value
        result = values[self.plan.output_ref]
        self._check(result, self.plan.output_shape, batch, "output")
        state_names = tuple(name for name, _ in self.named_parameters()) + tuple(name for name, _ in self.named_buffers())
        if state_names != self._state_names:
            raise HNDLError("E_RUNTIME", "A custom module created or removed registered state during forward")
        return result

    def forward(self, **inputs):
        if set(inputs) != {"x"}:
            raise HNDLError("E_BINDING", "Expected exactly the external tensor input 'x'")
        return {"output": self._execute(inputs["x"])}

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

    def __repr__(self):
        lines = [f"{type(self).__name__}: {_shape_text(self.plan.input_shape)} -> "
                 f"{_shape_text(self.plan.output_shape)}  dtype={self.plan.dtype}"]
        lines.append("index  name  operation  input shapes  output shapes")
        for i, node in enumerate(self.plan.nodes):
            inputs = ", ".join(f"{p}={node.inputs[p]}:{_shape_text(s)}" for p, s in node.input_shapes.items())
            outputs = ", ".join(f"{p}={_shape_text(s)}" for p, s in node.output_shapes.items())
            lines.append(f"{i}  {node.id}  {node.op}  {inputs}  {outputs}")
        return "\n".join(lines)


class Network(GraphModule):
    """Single-input, single-output tensor facade with shared graph state."""

    def forward(self, x):
        return self._execute(x)


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


def _builtin(node, device):
    args = dict(node.args)
    factory = {"device": device, "dtype": torch.float32}
    op = node.op
    if op == "linear@1":
        return nn.Linear(**args, **factory)
    if op == "conv2d@1":
        return nn.Conv2d(**args, **factory)
    if op == "conv_transpose2d@1":
        return nn.ConvTranspose2d(**args, **factory)
    if op == "group_norm@1":
        return nn.GroupNorm(**args, **factory)
    if op == "reshape@1":
        return _Reshape(args["shape"])
    if op == "flatten@1":
        return nn.Flatten(start_dim=1)
    if op == "relu@1":
        return nn.ReLU(inplace=False)
    if op == "leaky_relu@1":
        return nn.LeakyReLU(**args, inplace=False)
    if op == "tanh@1":
        return nn.Tanh()
    if op == "add@1":
        return _Add()
    if op == "split@1":
        dim, size = args["dim"], args["size"]
        return _Split(size, node.input_shapes["x"][dim] - size, dim)
    if op == "concat@1":
        return _Concat(args["axis"])
    return None


def _state_bytes(module):
    # Count unique storage, including nonpersistent registered buffers.
    storages = {}
    for value in (*module.parameters(), *module.buffers()):
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr())
        storages[key] = max(storages.get(key, 0), storage.nbytes())
    return sum(storages.values())


def _build(plan, *, device, initialization_seed=None, registry=None, facade=False):
    if plan.dtype != "float32":
        raise HNDLError("E_SCHEMA", f"Unsupported plan dtype {plan.dtype!r}")
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
    from .resolver import validate_concrete_plan
    plan = validate_concrete_plan(plan, registry=registry)
    # Validate all required implementations before constructing any modules.
    custom = {}
    port_orders = {}
    for node in plan.nodes:
        entry = registry.by_identity(node.op)
        port_orders[node.id] = (tuple(f"x{i}" for i in range(node.args["input_count"]))
                                if node.op == "concat@1" else entry.input_ports)
        if getattr(node, "state_version", entry.state_version) != entry.state_version:
            raise HNDLError("E_STATE_VERSION", f"{node.id}: incompatible state version")
        if node.op not in _BUILTINS:
            binding = registry.backends.get(node.op)
            if binding is None:
                raise HNDLError("E_REGISTRY", f"No explicitly registered PyTorch backend for {node.op}")
            if binding.state_version != entry.state_version:
                raise HNDLError("E_STATE_VERSION", f"{node.id}: incompatible backend state version")
            custom[node.id] = binding
    modules = OrderedDict()
    with _initialization_rng(device, initialization_seed):
        for node in plan.nodes:
            if node.id in custom:
                layer = custom[node.id].module(**dict(node.args))
                if not isinstance(layer, nn.Module):
                    raise HNDLError("E_REGISTRY", f"{node.id}: backend constructor must return nn.Module")
                if _state_bytes(layer) > node.state_bytes:
                    raise HNDLError("E_RESOURCE", f"{node.id}: registered parameter/buffer storage exceeds declared {node.state_bytes} bytes")
                layer = layer.to(device=device, dtype=torch.float32)
            else:
                layer = _builtin(node, device)
                if layer is None:
                    raise HNDLError("E_REGISTRY", f"No supported PyTorch implementation for {node.op}")
            modules[f"n_{node.id}"] = layer
    receipt = {"torch_version": str(torch.__version__), "device": str(device),
               "initialization_seed": initialization_seed,
               "seed_mode": "caller" if initialization_seed is None else "isolated"}
    cls = Network if facade else GraphModule
    return cls(plan, modules, device, receipt, port_orders)


def build(plan, *, device, initialization_seed=None, registry=None):
    """Build a concrete plan; forward accepts named inputs and returns a dict."""
    return _build(plan, device=device, initialization_seed=initialization_seed, registry=registry)


def _network(resolve_name, source, *, input_shape, output_shape, device,
             dtype="float32", registry=None, initialization_seed=None, limits=None):
    import hndl
    resolve = getattr(hndl, resolve_name)
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape,
                   dtype=dtype, registry=registry, limits=limits)
    return _build(plan, device=device, initialization_seed=initialization_seed,
                  registry=registry, facade=True)


def network(source, *, input_shape, output_shape, device, dtype="float32",
            registry=None, initialization_seed=None, limits=None):
    """Resolve declarative source and return a tensor-in/tensor-out module."""
    return _network("resolve", source, input_shape=input_shape, output_shape=output_shape,
                    device=device, dtype=dtype, registry=registry,
                    initialization_seed=initialization_seed, limits=limits)


def network_file(path, *, input_shape, output_shape, device, dtype="float32",
                 registry=None, initialization_seed=None, limits=None):
    """Read declarative UTF-8 source and construct its resolved network."""
    return _network("resolve_file", path, input_shape=input_shape, output_shape=output_shape,
                    device=device, dtype=dtype, registry=registry,
                    initialization_seed=initialization_seed, limits=limits)


def network_from_callable(fn, *, input_shape, output_shape, device, dtype="float32",
                          registry=None, initialization_seed=None, limits=None):
    """Capture trusted Python exactly once, then construct the resolved network."""
    return _network("resolve_callable", fn, input_shape=input_shape, output_shape=output_shape,
                    device=device, dtype=dtype, registry=registry,
                    initialization_seed=initialization_seed, limits=limits)
