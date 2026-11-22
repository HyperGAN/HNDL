"""Prepared execution of resolved graphs, sharing their existing parameters."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from functools import lru_cache
import itertools
import threading
import time
import weakref

import torch
from torch import nn
from torch.nn.modules import module as _torch_module
from torch.utils.checkpoint import checkpoint

from .errors import HNDLError
from . import torch as _runtime


_Tensor = torch.Tensor
_chain = itertools.chain.from_iterable
_get_tracing_state = torch._C._get_tracing_state
_BUILTIN_TENSOR_MODULES = frozenset({
    nn.Linear, nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.ELU,
    nn.LeakyReLU, nn.Hardswish, nn.Mish, nn.Softplus, nn.Identity,
    nn.Dropout, nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d, nn.BatchNorm2d,
    nn.GroupNorm, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
    nn.ConvTranspose2d, nn.Flatten, nn.Softmax,
})
_CAPTURE_LOCK = threading.RLock()
_CAPTURE_STREAMS = {}


def _release_graphs(graphs):
    """Keep replay storage alive until queued graph work and copies finish."""
    for entry in graphs.values():
        entry[4].synchronize()
    graphs.clear()


def _one(result, port):
    if type(result) is torch.Tensor:
        return result
    if isinstance(result, dict):
        if result.keys() != {port}:
            raise HNDLError("E_RUNTIME", f"expected exactly the output port {port!r}")
        return result[port]
    if isinstance(result, (tuple, list)):
        if len(result) != 1:
            raise HNDLError("E_RUNTIME", "expected one output tensor")
        return result[0]
    return result


def _many(result, ports):
    if isinstance(result, dict):
        if result.keys() != set(ports):
            raise HNDLError("E_RUNTIME", f"expected exactly the output ports {ports!r}")
        return tuple(result[port] for port in ports)
    if not isinstance(result, (tuple, list)) or len(result) != len(ports):
        raise HNDLError("E_RUNTIME", f"expected the output ports {ports!r}")
    return result


class _ExecutionModule(nn.Module):
    def __init__(self, nodes, training):
        super().__init__()
        self.nodes = nodes
        layers = tuple(nodes.values())
        self._layers = layers
        self.training = training
        hook_dicts = [
            _torch_module._global_forward_hooks,
            _torch_module._global_forward_pre_hooks,
            _torch_module._global_backward_hooks,
            _torch_module._global_backward_pre_hooks,
        ]
        can_direct = True
        for layer in layers:
            hook_dicts.extend((
                layer._forward_hooks,
                layer._forward_pre_hooks,
                layer._backward_hooks,
                layer._backward_pre_hooks,
            ))
            if (type(layer).__call__ is not nn.Module.__call__
                    or "__call__" in layer.__dict__
                    or getattr(layer, "_compiled_call_impl", None) is not None):
                can_direct = False
        self._hook_dicts = tuple(hook_dicts)
        self._can_direct_forward = can_direct


@lru_cache(maxsize=128)
def _executor_class(source):
    # Only compiler-owned identifiers and repr-quoted port names appear here.
    # Configuration source is never executed as Python.
    namespace = {"_one": _one, "_many": _many, "_checkpoint": checkpoint,
                 "_Tensor": _Tensor, "torch": torch}
    exec(compile(source, "<hndl prepared graph>", "exec"), namespace)
    return type("PreparedGraph", (_ExecutionModule,),
                {"forward": namespace["forward"],
                 "_forward_fast": namespace["_forward_fast"],
                 "__module__": __name__})


def _emit_forward(model, *, cached, fn_name, direct):
    refs = {f"input:{name}": f"i{index}" for index, name in enumerate(model._input_names)}
    arguments = ", ".join(refs.values())
    lines = [f"def {fn_name}(self, {'cache, ' if cached else ''}{arguments}):"]
    node_count = len(model.plan.nodes)
    if node_count == 1:
        lines.append("    l0, = self._layers")
    elif node_count > 1:
        lines.append("    " + ", ".join(f"l{index}" for index in range(node_count)) + " = self._layers")
    output_refs = {entry["ref"] for entry in model.plan.outputs.values()}
    last_use = {}
    for index, node in enumerate(model.plan.nodes):
        for ref in node.inputs.values():
            last_use[ref] = index
    for index, node in enumerate(model.plan.nodes):
        bound = [refs[node.inputs[port]] for port in model._port_orders[node.id]]
        layer_var = f"l{index}"
        callee = f"{layer_var}.forward" if direct else layer_var
        call = f"{callee}({', '.join(bound)})"
        if cached and getattr(model.nodes["n_" + node.id], "_supports_inference_cache", False):
            positional = (", ".join(bound) + ", ") if bound else ""
            call = f"{callee}({positional}cache=cache, node_id={node.id!r})"
        elif not cached and node.id in model.checkpoint_nodes:
            checkpoint_args = (", " + ", ".join(bound)) if bound else ""
            call = (f"(_checkpoint({layer_var}{checkpoint_args}, use_reentrant=False) "
                    f"if self.training and torch.is_grad_enabled() else {call})")
        variables = []
        for port_index, port in enumerate(node.outputs):
            variable = f"v{index}_{port_index}"
            refs[f"node:{node.id}/{port}"] = variable
            variables.append(variable)
        if len(variables) == 1:
            v = variables[0]
            lines.append(f"    {v} = {call}")
            layer_obj = model.nodes["n_" + node.id]
            guaranteed_tensor = (
                direct
                and type(layer_obj) in _BUILTIN_TENSOR_MODULES
                and "forward" not in layer_obj.__dict__
            )
            if not guaranteed_tensor:
                lines.append(f"    if type({v}) is not _Tensor:")
                lines.append(f"        {v} = _one({v}, {node.outputs[0]!r})")
        else:
            lines.append(f"    {', '.join(variables)} = _many({call}, {tuple(node.outputs)!r})")
        dead = {ref for ref in node.inputs.values()
                if last_use.get(ref) == index and ref not in output_refs and ref.startswith("node:")}
        dead.update(f"node:{node.id}/{port}" for port in node.outputs
                    if f"node:{node.id}/{port}" not in last_use
                    and f"node:{node.id}/{port}" not in output_refs)
        if dead:
            lines.append("    del " + ", ".join(refs[ref] for ref in sorted(dead)))
    outputs = [(name, refs[entry["ref"]]) for name, entry in model.plan.outputs.items()]
    if len(outputs) == 1:
        lines.append(f"    return {outputs[0][1]}")
    else:
        lines.append("    return {" + ", ".join(f"{name!r}: {value}" for name, value in outputs) + "}")
    return "\n".join(lines) + "\n"


def execution_module(model, *, cached=False):
    """Generate a straight-line module with the source graph's registered nodes."""
    checked_src = _emit_forward(model, cached=cached, fn_name="forward", direct=False)
    fast_src = _emit_forward(model, cached=cached, fn_name="_forward_fast", direct=True)
    source = checked_src + "\n" + fast_src
    result = _executor_class(source)(model.nodes, model.training)
    result.source_code = checked_src
    return result


def _signature(inputs):
    return tuple((tuple(value.shape), tuple(value.stride()), value.dtype, value.device, value.requires_grad)
                 if isinstance(value, torch.Tensor) else None for value in inputs)


def _clone_output(result):
    if isinstance(result, dict):
        memo = {}
        output = {}
        for name, value in result.items():
            if id(value) not in memo:
                memo[id(value)] = value.clone()
            output[name] = memo[id(value)]
        return output
    return result.clone()


def _captured_tensors(model):
    """Pin tensor storage in derived tables as well as registered model state."""
    result, seen = [], set()

    def collect(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            result.append(value.detach())
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                collect(item)
        elif not isinstance(value, nn.Module) and not callable(value) and hasattr(value, "__dict__"):
            collect(vars(value))

    collect(model._parameters)
    collect(model._buffers)
    for module in model.nodes.modules():
        collect(vars(module))
    return tuple(result)


class PreparedNetwork(nn.Module):
    """A generated executor sharing a graph's layers, hooks and optimizer state.

    Preparation fixes the module structure. Call ``source.rebind_state()`` after
    direct edits to registered state; moving/casting the source invalidates
    preparations automatically. Ordinary in-place optimizer updates work.

    ``cuda_graph`` requires eval mode with gradients and autocast disabled.
    Its LRU cache is bounded by entry count and retained CUDA allocations.
    Outputs are cloned by default; ``clone_outputs=False`` returns borrowed
    buffers overwritten by a later replay of the same entry.
    """

    def __setattr__(self, name, value):
        if name in {"source", "nodes"} and hasattr(self, name):
            raise TypeError("HNDL architecture is fixed; mutate and rebind the source graph")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in {"source", "nodes"}:
            raise TypeError("HNDL architecture is fixed; mutate and rebind the source graph")
        super().__delattr__(name)

    def __init__(self, model, example_inputs=None, *, backend="generated", max_graphs=4,
                 max_graph_bytes=256 * 1024 * 1024, clone_outputs=True, pack=False,
                 attention_backend="reference"):
        super().__init__()
        if backend not in {"generated", "cuda_graph", "auto"}:
            raise ValueError("backend must be 'generated', 'cuda_graph' or 'auto'")
        if type(max_graphs) is not int or max_graphs <= 0:
            raise ValueError("max_graphs must be a positive integer")
        if type(max_graph_bytes) is not int or max_graph_bytes <= 0:
            raise ValueError("max_graph_bytes must be a positive integer")
        self._packing_enabled = bool(pack)
        if pack:
            from .operators._packing import prepare_packed
            model.rebind_state()  # Finish replay before moving parameter storage.
            prepare_packed(model, trusted=True)
            model.rebind_state()
        if attention_backend != "reference":
            from .operators._attention_backend import prepare_attention
            model.rebind_state()  # Release captures before resetting cached masks.
            prepare_attention(model, backend=attention_backend)
        # Register the existing node tree at its original path, so checkpoints,
        # optimizers and enclosing PyTorch modules keep ordinary state names.
        object.__setattr__(self, "source", model)
        self.nodes = model.nodes
        self._parameters = model._parameters
        self._buffers = model._buffers
        self.backend = backend
        self.max_graphs = max_graphs
        self.max_graph_bytes = max_graph_bytes
        self.clone_outputs = clone_outputs
        self._graphs = OrderedDict()
        # The callback retains the buffers, never this wrapper. In particular,
        # side-stream replay must finish before normal GC releases input or
        # captured-state storage back to its original allocation stream.
        self._graph_finalizer = weakref.finalize(self, _release_graphs, self._graphs)
        self._graph_bytes = 0
        self._validated = OrderedDict()
        self._validated_keys = frozenset()
        self._last_fast = None
        self._choices = OrderedDict()
        self._lock = threading.RLock()
        self.training = model.training
        self._always_single = isinstance(model, _runtime.Network) and model._single_output
        self._single_input = len(model._input_names) == 1
        self._is_generated = backend == "generated"
        self._source_single_output = model._single_output
        self._refresh()
        model._prepared_dependents.add(self)
        if example_inputs is not None:
            if isinstance(example_inputs, Mapping):
                self(**example_inputs)
            else:
                self(*tuple(example_inputs))

    def _refresh(self):
        with self._lock:
            self.clear_cache()
            self.training = self.source.training
            self.source._verify_state("build")
            if self._packing_enabled:
                from .operators._packing import prepare_packed
                prepare_packed(self.source, trusted=True)
            object.__setattr__(self, "_export_module", execution_module(self.source))
            self._epoch = getattr(self.source, "_preparation_epoch", 0)
            self._generation = _runtime._watch_generation

    def clear_cache(self):
        """Release retained replay buffers and forget validated signatures."""
        with self._lock:
            for entry in self._graphs.values():
                entry[4].synchronize()
            self._graphs.clear()
            self._graph_bytes = 0
            self._validated.clear()
            self._validated_keys = frozenset()
            self._last_fast = None
            self._choices.clear()

    def _make_fast_entry(self, generation, single_tensor):
        export_mod = self._export_module
        if not (self._is_generated and self._single_input and export_mod._can_direct_forward):
            return None
        source = self.source
        same, copies, views, store_lens, tensor_ids, _ = source._stores
        all_hooks = (
            source._forward_hooks,
            source._forward_pre_hooks,
            source._backward_hooks,
            *export_mod._hook_dicts,
        )
        return (
            source,
            self._epoch,
            generation,
            source.training,
            single_tensor.shape,
            single_tensor.dtype,
            single_tensor.device,
            export_mod._forward_fast,
            all_hooks,
            same,
            copies,
            views,
            store_lens,
            tensor_ids,
        )

    def _record_validated(self, key, single_tensor=None):
        with self._lock:
            if (self._epoch == self.source._preparation_epoch
                    and self._generation == _runtime._watch_generation):
                self._validated[key] = True
                if len(self._validated) > 64:
                    self._validated.popitem(last=False)
                self._validated_keys = frozenset(self._validated)
                if single_tensor is not None and not _runtime._autocast_enabled():
                    self._last_fast = self._make_fast_entry(self._generation, single_tensor)

    def __deepcopy__(self, memo):
        result = type(self)(copy.deepcopy(self.source, memo), backend=self.backend,
                            max_graphs=self.max_graphs, max_graph_bytes=self.max_graph_bytes,
                            clone_outputs=self.clone_outputs, pack=self._packing_enabled)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if "hook" in name:
                result.__dict__[name] = copy.deepcopy(value, memo)
        return result

    def __copy__(self):
        result = type(self)(self.source, backend=self.backend,
                            max_graphs=self.max_graphs, max_graph_bytes=self.max_graph_bytes,
                            clone_outputs=self.clone_outputs)
        result._packing_enabled = self._packing_enabled
        return result

    def _evict_graph(self, key=None):
        entry = self._graphs.pop(key) if key is not None else self._graphs.popitem(last=False)[1]
        entry[4].synchronize()
        self._graph_bytes -= entry[3]

    @property
    def cache_info(self):
        return {"graphs": len(self._graphs), "bytes": self._graph_bytes,
                "max_graphs": self.max_graphs, "max_bytes": self.max_graph_bytes,
                "strategies": dict(self._choices)}

    @property
    def plan(self):
        return self.source.plan

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._refresh()
        return result

    def __getitem__(self, index):
        return self.source[index]

    def param_groups(self, weight_decay=0.01):
        return self.source.param_groups(weight_decay)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        self.source._apply(fn, recurse=recurse)
        return self

    def train(self, mode=True):
        if self.training != mode:
            self.clear_cache()
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.source.train(mode)
        return self

    def _graph(self, inputs, key):
        if key in self._graphs:
            entry = self._graphs.pop(key)
            self._graphs[key] = entry
            return entry
        self._capture_supported(inputs)
        while self._graphs and len(self._graphs) >= self.max_graphs:
            self._evict_graph()
        device = inputs[0].device
        with _CAPTURE_LOCK, torch.cuda.device(device):
            before = torch.cuda.memory_allocated(device)
            # Preserve strides, including non-contiguous tensor inputs.
            static_inputs = tuple(torch.empty_strided(v.shape, v.stride(), dtype=v.dtype,
                                                      device=v.device).copy_(v) for v in inputs)
            stream = _CAPTURE_STREAMS.get(device.index)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                _CAPTURE_STREAMS[device.index] = stream
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._export_module(*static_inputs)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self._export_module(*static_inputs)
            # Captured intermediates remain reserved in the graph's private
            # pool even after their tensors die. Allocated-byte deltas alone
            # substantially undercount this retained workspace.
            segments = [segment for segment in torch.cuda.memory_snapshot()
                        if segment["device"] == device.index
                        and tuple(segment.get("segment_pool_id", ())) == tuple(graph.pool())]
            input_bytes = sum(v.untyped_storage().nbytes() for v in static_inputs)
            captured_state = _captured_tensors(self.source)
            registered = {(v.device, v.untyped_storage().data_ptr()) for v in (
                *self.source.parameters(), *self.source.buffers())}
            derived = {}
            for value in captured_state:
                pointer = value.untyped_storage().data_ptr()
                storage_key = (value.device, pointer)
                if (value.device == device and storage_key not in registered
                        and not any(segment["address"] <= pointer < segment["address"] + segment["total_size"]
                                    for segment in segments)):
                    derived[storage_key] = value.untyped_storage().nbytes()
            size = (sum(segment["total_size"] for segment in segments) + input_bytes
                    if segments else max(torch.cuda.memory_reserved(device),
                                         torch.cuda.memory_allocated(device) - before))
            size += sum(derived.values())
        if size > self.max_graph_bytes:
            raise HNDLError("E_RESOURCE", f"CUDA graph needs {size} bytes; max_graph_bytes={self.max_graph_bytes}")
        while self._graphs and self._graph_bytes + size > self.max_graph_bytes:
            self._evict_graph()
        # Keep the actual captured storage alive through in-flight replay, even
        # if a Parameter is replaced or its .data storage is reassigned.
        entry = (graph, static_inputs, outputs, size, torch.cuda.Event(), captured_state)
        self._graphs[key] = entry
        self._graph_bytes += size
        return entry

    def _capture_supported(self, inputs):
        for node in self.source.plan.nodes:
            layer = self.source.nodes["n_" + node.id]
            builtin = type(layer).__module__.startswith(("hndl.operators.", "torch.nn."))
            pretrained = self.source.plan.registry.by_identity(node.op).alias == "pretrained"
            if not getattr(layer, "_hndl_cuda_graph_safe", builtin and not pretrained):
                raise HNDLError("E_UNSUPPORTED", f"CUDA capture requires an explicit _hndl_cuda_graph_safe declaration for {node.id!r}")
        if any(layer._forward_hooks or layer._forward_pre_hooks for layer in self.source.modules()):
            raise HNDLError("E_UNSUPPORTED", "CUDA replay cannot execute Python forward hooks; use generated execution")
        storage = sum((1 + sum((size - 1) * stride for size, stride in zip(v.shape, v.stride())))
                      * v.element_size() for v in inputs)
        if storage > self.max_graph_bytes:
            raise HNDLError("E_RESOURCE", f"CUDA graph input buffers exceed max_graph_bytes={self.max_graph_bytes}")
        if any(any(stride == 0 and size > 1 for size, stride in zip(v.shape, v.stride())) for v in inputs):
            raise HNDLError("E_UNSUPPORTED", "CUDA replay input buffers require non-overlapping strides")

    def _run_graph(self, inputs, key):
        if self.source.training or torch.is_grad_enabled() or any(v.requires_grad for v in inputs):
            raise HNDLError("E_RUNTIME", "CUDA graph execution requires eval() and no_grad() or inference_mode()")
        from .torch import _autocast_enabled
        if _autocast_enabled():
            raise HNDLError("E_RUNTIME", "CUDA graph preparation requires autocast disabled; cast the model explicitly")
        if not inputs or any(v.device.type != "cuda" or v.device != inputs[0].device for v in inputs):
            raise HNDLError("E_RUNTIME", "CUDA graph execution requires all inputs on one CUDA device")
        graph, buffers, outputs, _, completed, _ = self._graph(inputs, key)
        current_stream = torch.cuda.current_stream(inputs[0].device)
        current_stream.wait_event(completed)
        for buffer, value in zip(buffers, inputs):
            buffer.copy_(value)
        graph.replay()
        result = _clone_output(outputs) if self.clone_outputs else outputs
        completed.record(current_stream)
        return result

    def _run_auto(self, inputs, key, checked=None):
        choice = self._choices.get(key)
        if choice is None:
            choice = "generated"
            eligible = (not self.source.training and not torch.is_grad_enabled()
                        and inputs and all(isinstance(v, torch.Tensor) and v.device.type == "cuda"
                                           and not v.requires_grad for v in inputs))
            if eligible:
                try:
                    self._run_graph(inputs, key)
                    candidates = {"generated": lambda: self._export_module(*inputs),
                                  "cuda_graph": lambda: self._run_graph(inputs, key)}
                    timings = {name: [] for name in candidates}
                    for round_index in range(3):
                        names = tuple(candidates) if round_index % 2 == 0 else tuple(reversed(candidates))
                        for name in names:
                            torch.cuda.synchronize(inputs[0].device)
                            start = time.perf_counter()
                            for _ in range(20):
                                candidates[name]()
                            torch.cuda.synchronize(inputs[0].device)
                            timings[name].append(time.perf_counter() - start)
                    if min(timings["cuda_graph"]) < min(timings["generated"]):
                        choice = "cuda_graph"
                    else:
                        self._evict_graph(key)
                except (HNDLError, RuntimeError):
                    # Auto is an optional optimization; unsupported capture or
                    # a strict memory budget preserves generated execution.
                    if key in self._graphs:
                        self._evict_graph(key)
            self._choices[key] = choice
            if len(self._choices) > 64:
                self._choices.popitem(last=False)
        if choice == "cuda_graph":
            return self._run_graph(inputs, key)
        return checked if checked is not None else self._export_module(*inputs)

    def forward(self, *args, cache=None, **named):
        last = self._last_fast
        if last is not None and cache is None and not named and len(args) == 1:
            x = args[0]
            (source, epoch, generation, training, shape, dtype, device,
             fast_fn, all_hooks, same, copies, views, store_lens, tensor_ids) = last
            try:
                fast_ok = (
                    type(x) is _Tensor
                    and x.shape == shape
                    and x.dtype is dtype
                    and x.device == device
                    and epoch == source._preparation_epoch
                    and generation == _runtime._watch_generation
                    and training == source.training
                    and not _runtime._autocast_enabled()
                    and not _runtime._is_compiling()
                    and not _get_tracing_state()
                    and not any(all_hooks)
                    and same == copies
                    and tuple(map(len, views)) == store_lens
                    and tuple(map(id, _chain(views))) == tensor_ids
                )
            except Exception:
                fast_ok = False
            if fast_ok:
                result = fast_fn(x)
                if _runtime._watch_generation != generation:
                    new_gen = _runtime._watch_generation
                    source._verify_state("forward")
                    source._state_checked(new_gen)
                    self._refresh()
                return result if self._always_single else self._format(result, args, named)
        if self.backend != "cuda_graph" and (self.source._forward_hooks
                or self.source._forward_pre_hooks or self.source._backward_hooks):
            # Root hooks belong to the original module; let PyTorch invoke
            # them with that module and its normal input/output conventions.
            return self.source(*args, cache=cache, **named)
        if cache is not None and "cache" in self.source._input_set:
            cache = self.source._bind_cache_port(args, named, cache)
        inputs = self.source._bind(args, named)
        if _runtime._is_compiling():
            if cache is not None or self.backend == "cuda_graph":
                raise HNDLError("E_UNSUPPORTED", "Compile the generated executor without a stateful cache or CUDA graph wrapper")
            if self._epoch != self.source._preparation_epoch:
                raise HNDLError("E_RUNTIME", "prepared executor changed; call source.rebind_state() before compiled execution")
            # Dynamo specializes these port checks as guards during tracing.
            # The raw export module remains available for unchecked tensor programs.
            outputs = self.source._run_checked(inputs, compiling=True)
            if self.source._single_output:
                return self._format(next(iter(outputs.values())), args, named, compiling=True)
            return outputs
        if self.backend == "generated" and cache is None:
            if (self._epoch != self.source._preparation_epoch
                    or self._generation != _runtime._watch_generation
                    or self.source._stores_changed()):
                self._refresh()
            generation = self._generation
            key = self.source._signature(inputs)
            if key not in self._validated_keys:
                result = self.source._execute(inputs, single=self._source_single_output)
                single_tensor = inputs[0] if (self._single_input and type(inputs[0]) is _Tensor) else None
                self._record_validated(key, single_tensor)
            else:
                export_mod = self._export_module
                if (export_mod._can_direct_forward
                        and not any(export_mod._hook_dicts)
                        and not _get_tracing_state()):
                    result = export_mod._forward_fast(*inputs)
                else:
                    result = export_mod.forward(*inputs)
                if _runtime._watch_generation != generation:
                    new_gen = _runtime._watch_generation
                    self.source._verify_state("forward")
                    self.source._state_checked(new_gen)
                    self._refresh()
                elif (self._single_input and not _runtime._autocast_enabled()
                        and type(inputs[0]) is _Tensor):
                    self._last_fast = self._make_fast_entry(generation, inputs[0])
            return result if self._always_single else self._format(result, args, named)
        with self._lock:
            if (self._epoch != getattr(self.source, "_preparation_epoch", 0)
                    or self._generation != _runtime._watch_generation
                    or self.source._stores_changed()):
                self._refresh()
            if cache is not None:
                if self.backend == "cuda_graph":
                    raise HNDLError("E_UNSUPPORTED", "stateful decoding uses generated execution, not CUDA replay")
                return self.source(*args, cache=cache, **named)
            key = (self.source.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                   _runtime._autocast_enabled(), _signature(inputs))
            checked = None
            if key not in self._validated:
                # Execute the checked path once, retaining its rich port errors.
                checked = self.source._execute(inputs, single=self._source_single_output)
                self._validated[key] = True
                if len(self._validated) > 64:
                    self._validated.popitem(last=False)
                self._validated_keys = frozenset(self._validated)
                if self.backend == "generated":
                    return self._format(checked, args, named)
            if self.backend == "cuda_graph":
                result = self._run_graph(inputs, key)
            elif self.backend == "auto":
                result = self._run_auto(inputs, key, checked)
            else:
                result = self._export_module.forward(*inputs)
            return self._format(result, args, named)

    def _format(self, result, args, named, compiling=False):
        if self.source._single_output:
            if self._always_single or (args and not named):
                return result
            result = {next(iter(self.source.plan.outputs)): result}
        return result if compiling else _runtime.OutputDict(result)
