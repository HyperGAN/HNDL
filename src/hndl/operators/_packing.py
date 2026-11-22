"""Optional packed storage that preserves public parameters and checkpoint keys.

Packing rebinds the storage of existing Parameter objects, never replaces them.
Optimizer updates therefore reach the packed view immediately. The autograd
bridge distributes a packed gradient to the original public parameters.
"""

from collections import Counter
import sys
import weakref

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import module as _torch_module

_functorch_active = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)


class _PackedParameters(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, sizes, stacked, *parameters):
        ctx.sizes = sizes
        ctx.stacked = stacked
        return packed.view_as(packed)

    @staticmethod
    def backward(ctx, gradient):
        pieces = gradient.unbind(0) if ctx.stacked else gradient.split(ctx.sizes, dim=0)
        return None, None, None, *pieces


class _PackedExpertGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, sizes, num_experts, *parameters):
        ctx.sizes = sizes
        ctx.num_experts = num_experts
        return packed.view_as(packed)

    @staticmethod
    def backward(ctx, gradient):
        unbound = [chunk.unbind(0) for chunk in gradient.split(ctx.sizes, dim=1)]
        pieces = [unbound[j][e] for e in range(ctx.num_experts) for j in range(len(unbound))]
        return None, None, None, *pieces


class _VersionGuard(torch.autograd.Function):
    """Tie input-gradient execution to the original weights' version counters.

    Packed storage aliases Parameter data but has its own Tensor version
    counter. Saving the public weights restores ordinary in-place mutation
    checks, including when all those weights are frozen.
    """
    @staticmethod
    def forward(ctx, inputs, *parameters):
        ctx.save_for_backward(*parameters)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        parameters = ctx.saved_tensors
        return gradient, *(None for _ in parameters)


def _signature(parameters):
    return tuple((id(p), p.data_ptr(), p.shape, p.is_contiguous(), p.dtype, p.device) for p in parameters)


def _trust_token(owner):
    # Deepcopies retain weakrefs to the original root, so do not trust copied
    # metadata until preparation explicitly binds it to the copied module.
    if getattr(owner, "_hndl_packing_module_id", None) != id(owner):
        return None
    reference = getattr(owner, "_hndl_packing_owner", None)
    root = None if reference is None else reference()
    runtime = sys.modules.get("hndl.torch")
    if root is None or runtime is None:
        return None
    return (id(root), getattr(root, "_preparation_epoch", 0), runtime._watch_generation)


class _Layout:
    def __init__(self, parameters, stacked, bindings, *, group_sizes=None):
        self.parameters = tuple(parameters)
        self.bindings = bindings
        self.stacked = stacked
        self.group_sizes = tuple(group_sizes) if group_sizes is not None else None
        self.num_experts = (len(self.parameters) // len(self.group_sizes)
                            if self.group_sizes is not None else 0)
        self.sizes = tuple(p.shape[0] for p in self.parameters)
        # Normal tensors also work when preparation takes place inside
        # inference_mode and this same model is subsequently trained.
        with torch.inference_mode(False), torch.no_grad():
            if self.group_sizes is not None:
                per_expert = len(self.group_sizes)
                packed = torch.stack([
                    torch.cat(self.parameters[e * per_expert:(e + 1) * per_expert], dim=0)
                    for e in range(self.num_experts)
                ], dim=0).detach()
                for e in range(self.num_experts):
                    views = packed[e].split(self.group_sizes, dim=0)
                    for j, view in enumerate(views):
                        self.parameters[e * per_expert + j].data = view
            else:
                packed = (torch.stack(self.parameters) if stacked else torch.cat(self.parameters, dim=0)).detach()
                views = packed.unbind(0) if stacked else packed.split(self.sizes, dim=0)
                for parameter, view in zip(self.parameters, views):
                    parameter.data = view
        self.packed = packed
        self.signature = _signature(self.parameters)
        self._storage_checks = tuple((p, p.data_ptr(), p.shape, p.dtype) for p in self.parameters)
        self.token = None

    def matches(self, parameters):
        return self.signature == _signature(parameters)

    def identities_match(self):
        return all(store.get(name) is parameter for store, name, parameter in self.bindings)

    def storage_matches(self):
        for p, ptr, shape, dtype in self._storage_checks:
            if p.data_ptr() != ptr or p.shape != shape or p.dtype is not dtype or not p.is_contiguous():
                return False
        return True

    def value(self):
        if not torch.is_grad_enabled():
            return self.packed
        if self.group_sizes is not None:
            return _PackedExpertGroup.apply(self.packed, self.group_sizes, self.num_experts, *self.parameters)
        return _PackedParameters.apply(self.packed, self.sizes, self.stacked, *self.parameters)


def _bindings(owner, key):
    if key[0] == "experts":
        _, name, attribute = key
        modules = (expert._modules[name] for expert in owner.experts)
        return tuple((module._parameters, attribute, module._parameters[attribute]) for module in modules)
    if key[0] == "experts_group":
        _, names, attribute = key
        return tuple(
            (expert._modules[name]._parameters, attribute, expert._modules[name]._parameters[attribute])
            for expert in owner.experts
            for name in names
        )
    names, attribute = key
    modules = (getattr(owner, name) for name in names)
    return tuple((module._parameters, attribute, module._parameters[attribute]) for module in modules)


def _fallback_pack(parameters, stacked, group_sizes):
    if group_sizes is not None:
        per_expert = len(group_sizes)
        num_experts = len(parameters) // per_expert
        return torch.stack([
            torch.cat(parameters[e * per_expert:(e + 1) * per_expert], dim=0)
            for e in range(num_experts)
        ], dim=0)
    return torch.stack(parameters) if stacked else torch.cat(parameters, dim=0)


def _packed(owner, key, parameters, *, stacked=False, group_sizes=None):
    if torch.compiler.is_compiling() or _functorch_active():
        return _fallback_pack(tuple(parameters), stacked, group_sizes)
    layouts = owner.__dict__.setdefault("_hndl_packed_layouts", {})
    token = _trust_token(owner)
    layout = layouts.get(key)
    # Stateless calls swap dictionary values without registration hooks. Keep
    # this cheap identity guard even under a trusted storage/shape contract,
    # and never rebind ephemeral functional parameters into cached storage.
    if layout is not None and not layout.identities_match():
        return _fallback_pack(tuple(parameters), stacked, group_sizes)
    if token is not None and layout is not None and layout.token == token:
        if layout.storage_matches():
            return layout.value()
        return _fallback_pack(tuple(parameters), stacked, group_sizes)
    parameters = tuple(parameters)
    blocked = getattr(owner, "_hndl_packing_blocked", ())
    if (len({id(p) for p in parameters}) != len(parameters)
            or any(id(p) in blocked or type(p) is not nn.Parameter for p in parameters)):
        return _fallback_pack(parameters, stacked, group_sizes)
    if layout is None or not layout.matches(parameters):
        layout = layouts[key] = _Layout(parameters, stacked, _bindings(owner, key), group_sizes=group_sizes)
    layout.token = token
    return layout.value()


def _has_call_hooks(module):
    return (module._forward_hooks or module._forward_pre_hooks
            or module._backward_hooks or module._backward_pre_hooks)


def _global_call_hooks():
    return (_torch_module._global_forward_hooks or _torch_module._global_forward_pre_hooks
            or _torch_module._global_backward_hooks or _torch_module._global_backward_pre_hooks)


def _plain_linear(module):
    return (type(module) is nn.Linear and "forward" not in module.__dict__
            and not _has_call_hooks(module) and not _global_call_hooks())


def project(owner, x, names):
    """Evaluate projections of one input with one GEMM when packing is enabled."""
    modules = tuple(getattr(owner, name) for name in names)
    if not getattr(owner, "_hndl_packing", False) or not all(_plain_linear(m) for m in modules):
        return tuple(module(x) for module in modules)
    biases = tuple(module.bias for module in modules)
    if any(b is None for b in biases) and not all(b is None for b in biases):
        return tuple(module(x) for module in modules)
    key = tuple(names)
    weight = _packed(owner, (key, "weight"), (m.weight for m in modules))
    bias = None if biases[0] is None else _packed(owner, (key, "bias"), biases)
    x = _guard_packed_input(owner, (key, "weight"), x, weight)
    return F.linear(x, weight, bias).split(tuple(m.out_features for m in modules), dim=-1)


def expert_parameters(owner, name, attribute="weight"):
    """An expert-major parameter view, with a differentiable unprepared path."""
    parameters = (expert._modules[name]._parameters[attribute] for expert in owner.experts)
    if getattr(owner, "_hndl_packing", False):
        return _packed(owner, ("experts", name, attribute), parameters, stacked=True)
    return torch.stack(tuple(parameters))


def expert_pair_parameters(owner, names, attribute="weight"):
    """An expert-major concatenated view ``[E, sum(H_i), ...]`` across projection names.

    Every underlying public parameter retains contiguous ``[H_i, ...]`` storage
    sharing the packed ``[E, sum(H_i), ...]`` buffer.
    """
    names = tuple(names)
    parameters = tuple(
        expert._modules[name]._parameters[attribute]
        for expert in owner.experts
        for name in names
    )
    if not owner.experts:
        return torch.empty(0)
    sizes = tuple(owner.experts[0]._modules[name]._parameters[attribute].shape[0] for name in names)
    if getattr(owner, "_hndl_packing", False):
        return _packed(owner, ("experts_group", names, attribute), parameters, group_sizes=sizes)
    return _fallback_pack(parameters, False, sizes)


def _guard_packed_input(owner, key, inputs, weights):
    if (torch.compiler.is_compiling() or _functorch_active() or not torch.is_grad_enabled()
            or not inputs.requires_grad or not getattr(owner, "_hndl_packing", False)):
        return inputs
    layout = getattr(owner, "_hndl_packed_layouts", {}).get(key)
    if layout is not None and layout.identities_match() and weights.data_ptr() == layout.packed.data_ptr():
        return _VersionGuard.apply(inputs, *layout.parameters)
    return inputs


def guard_expert_input(owner, name, inputs, weights):
    """Preserve version checks where an expert matrix aliases public weights."""
    return _guard_packed_input(owner, ("experts", name, "weight"), inputs, weights)


def guard_expert_pair_input(owner, names, inputs, weights):
    """Preserve version checks where a grouped expert matrix aliases public weights."""
    return _guard_packed_input(owner, ("experts_group", tuple(names), "weight"), inputs, weights)


def prepare_packed(module, *, trusted=False):
    """Enable packed projections recursively; keep parameter IDs and state keys.

    Tied parameters and distinct parameters sharing storage keep their existing
    layout. Hooks on individual projections continue to run via the plain path.
    """
    named = tuple(module.named_parameters(remove_duplicate=False))
    counts = Counter(id(p) for _, p in named)
    storage_ids = {}
    owned_storage = {}
    expert_children = set()
    for child in module.modules():
        for layout in getattr(child, "_hndl_packed_layouts", {}).values():
            key = (layout.packed.device, layout.packed.untyped_storage().data_ptr())
            owned_storage.setdefault(key, set()).update(id(p) for p in layout.parameters)
        if hasattr(child, "experts") and isinstance(child.experts, nn.ModuleList):
            expert_children.update(id(expert) for expert in child.experts)
    for _, parameter in named:
        key = (parameter.device, parameter.untyped_storage().data_ptr())
        storage_ids.setdefault(key, set()).add(id(parameter))
    blocked = {id(p) for _, p in named if counts[id(p)] > 1}
    blocked.update(identity for key, identities in storage_ids.items()
                   if len(identities) > 1 and not identities <= owned_storage.get(key, set())
                   for identity in identities)
    for child in module.modules():
        child._hndl_packing = True
        child._hndl_packing_blocked = frozenset(blocked)
        child.__dict__.setdefault("_hndl_packed_layouts", {})
        layouts = child._hndl_packed_layouts
        for key, layout in tuple(layouts.items()):
            if not layout.identities_match() or not layout.storage_matches():
                del layouts[key]
        if trusted:
            child._hndl_packing_owner = weakref.ref(module)
            child._hndl_packing_module_id = id(child)
        else:
            child.__dict__.pop("_hndl_packing_owner", None)
            child.__dict__.pop("_hndl_packing_module_id", None)
        groups = []
        if all(hasattr(child, name) for name in ("q_proj", "k_proj", "v_proj")):
            if getattr(child, "_packed_context_kv", False):
                groups.append(("k_proj", "v_proj"))
            elif child.q_proj.in_features == child.k_proj.in_features:
                groups.append(("q_proj", "k_proj", "v_proj"))
            else:
                groups.append(("k_proj", "v_proj"))
        if all(hasattr(child, name) for name in ("x_q_proj", "x_k_proj", "x_v_proj")):
            groups.append(("x_q_proj", "x_k_proj", "x_v_proj"))
        if all(hasattr(child, name) for name in ("c_q_proj", "c_k_proj", "c_v_proj")):
            groups.append(("c_q_proj", "c_k_proj", "c_v_proj"))
        if (hasattr(child, "k_up") and hasattr(child, "v_up")
                and not getattr(child, "is_weight_absorbed", False)):
            groups.append(("k_up", "v_up"))
        if id(child) not in expert_children and all(hasattr(child, name) for name in ("gate", "up")):
            groups.append(("gate", "up"))
        for names in groups:
            layers = tuple(getattr(child, name) for name in names)
            if not all(_plain_linear(layer) for layer in layers):
                continue
            _packed(child, (names, "weight"), (layer.weight for layer in layers))
            if all(layer.bias is not None for layer in layers):
                _packed(child, (names, "bias"), (layer.bias for layer in layers))
        if hasattr(child, "experts") and isinstance(child.experts, nn.ModuleList):
            child._hndl_grouped_mm = hasattr(torch, "_grouped_mm")
            for name in ("gate", "up", "down"):
                if child.experts and all(_plain_linear(getattr(expert, name, None)) for expert in child.experts):
                    expert_parameters(child, name)
                    if all(getattr(expert, name).bias is not None for expert in child.experts):
                        expert_parameters(child, name, "bias")
    return module


def clear_packed(module):
    """Disable packing and release metadata; public parameters retain their data."""
    for child in module.modules():
        child._hndl_packing = False
        child.__dict__.pop("_hndl_grouped_mm", None)
        child.__dict__.pop("_hndl_packed_layouts", None)
        child.__dict__.pop("_hndl_packing_blocked", None)
        child.__dict__.pop("_hndl_packing_owner", None)
        child.__dict__.pop("_hndl_packing_module_id", None)
    return module


def clear_packed_cache(module):
    """Release old packed views after a cast/move, retaining preparation flags."""
    for child in module.modules():
        child.__dict__.pop("_hndl_packed_layouts", None)
        child.__dict__.pop("_hndl_grouped_mm", None)
