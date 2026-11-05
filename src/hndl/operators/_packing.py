"""Optional packed storage that preserves public parameters and checkpoint keys.

Packing rebinds the storage of existing Parameter objects, never replaces them.
Optimizer updates therefore reach the packed view immediately. The autograd
bridge distributes a packed gradient to the original public parameters.
"""

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


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


def _signature(parameters):
    return tuple((id(p), p.data_ptr(), p.shape, p.stride(), p.dtype, p.device) for p in parameters)


class _Layout:
    def __init__(self, parameters, stacked):
        self.parameters = tuple(parameters)
        self.stacked = stacked
        self.sizes = tuple(p.shape[0] for p in parameters)
        # Normal tensors also work when preparation takes place inside
        # inference_mode and this same model is subsequently trained.
        with torch.inference_mode(False), torch.no_grad():
            packed = (torch.stack(parameters) if stacked else torch.cat(parameters, dim=0)).detach()
            views = packed.unbind(0) if stacked else packed.split(self.sizes, dim=0)
            for parameter, view in zip(parameters, views):
                parameter.data = view
        self.packed = packed
        self.signature = _signature(parameters)

    def matches(self, parameters):
        return self.signature == _signature(parameters)

    def value(self):
        return _PackedParameters.apply(self.packed, self.sizes, self.stacked, *self.parameters)


def _packed(owner, key, parameters, *, stacked=False):
    parameters = tuple(parameters)
    if torch.compiler.is_compiling():
        return torch.stack(parameters) if stacked else torch.cat(parameters, dim=0)
    blocked = getattr(owner, "_hndl_packing_blocked", ())
    if (len({id(p) for p in parameters}) != len(parameters)
            or any(id(p) in blocked or type(p) is not nn.Parameter for p in parameters)):
        return torch.stack(parameters) if stacked else torch.cat(parameters, dim=0)
    layouts = owner.__dict__.setdefault("_hndl_packed_layouts", {})
    layout = layouts.get(key)
    if layout is None or not layout.matches(parameters):
        layout = layouts[key] = _Layout(parameters, stacked)
    return layout.value()


def _plain_linear(module):
    return (type(module) is nn.Linear
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module._backward_hooks and not module._backward_pre_hooks)


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
    return F.linear(x, weight, bias).split(tuple(m.out_features for m in modules), dim=-1)


def expert_parameters(owner, name, attribute="weight"):
    """An expert-major parameter view, with a differentiable unprepared path."""
    parameters = tuple(getattr(getattr(expert, name), attribute) for expert in owner.experts)
    if getattr(owner, "_hndl_packing", False):
        return _packed(owner, ("experts", name, attribute), parameters, stacked=True)
    return torch.stack(parameters)


def prepare_packed(module):
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
        groups = []
        if all(hasattr(child, name) for name in ("q_proj", "k_proj", "v_proj")):
            if child.q_proj.in_features == child.k_proj.in_features:
                groups.append(("q_proj", "k_proj", "v_proj"))
            else:
                groups.append(("k_proj", "v_proj"))
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
            parameter = next(child.parameters(), None)
            child._hndl_grouped_mm = bool(
                hasattr(torch, "_grouped_mm") and parameter is not None
                and parameter.device.type == "cuda"
                and torch.cuda.get_device_capability(parameter.device)[0] >= 9
                and (child.d_model * parameter.element_size()) % 16 == 0
                and (child.hidden * parameter.element_size()) % 16 == 0)
            for name in ("gate", "up", "down"):
                if child.experts and all(hasattr(expert, name) for expert in child.experts):
                    expert_parameters(child, name)
                    if all(getattr(expert, name).bias is not None for expert in child.experts):
                        expert_parameters(child, name, "bias")
    return module


def clear_packed(module):
    """Disable packing and release metadata; public parameters retain their data."""
    for child in module.modules():
        child._hndl_packing = False
        child.__dict__.pop("_hndl_packed_layouts", None)
        child.__dict__.pop("_hndl_packing_blocked", None)
    return module
