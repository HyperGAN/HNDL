"""Standalone zero-dependency PyTorch code generation from ResolvedPlan."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from .operator import ELLIPSIS, Sym

if TYPE_CHECKING:
    from .types import ResolvedPlan


def _symbol_values(spec, node) -> dict[str, int]:
    """Resolved values of the shape symbols a module constructor requests."""
    values = {}
    shapes = {**node.input_shapes, **node.output_shapes}
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


def _get_constructor_kwargs(spec, node) -> dict[str, Any]:
    """Compute kwargs to instantiate spec.module for node."""
    kwargs = {name: arg.default for name, arg in spec.args.items() if arg.since is not None}
    kwargs.update(node.args)
    if spec.init_symbols:
        symbols = _symbol_values(spec, node)
        for name in spec.init_symbols:
            kwargs[name] = symbols[name]
    for name in spec.init_shapes:
        kwargs[name] = {port: tuple(shape) for port, shape in getattr(node, name).items()}
    return kwargs


def _format_value(val: Any) -> str:
    if isinstance(val, tuple):
        return "(" + ", ".join(_format_value(v) for v in val) + ("," if len(val) == 1 else "") + ")"
    if isinstance(val, list):
        return "[" + ", ".join(_format_value(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{repr(k)}: {_format_value(v)}" for k, v in val.items()) + "}"
    return repr(val)


def _format_module_constructor(spec, node, kwargs) -> tuple[str | None, str]:
    """Return (custom_class_code or None, constructor_call_string)."""
    alias = spec.alias
    if alias == "linear" and not kwargs.get("equalized") and not kwargs.get("spectral_norm"):
        in_f = kwargs["in_features"]
        out_f = kwargs["out_features"]
        bias = kwargs.get("bias", True)
        return None, f"nn.Linear({in_f}, {out_f}, bias={bias})"

    if alias == "relu":
        return None, "nn.ReLU()"

    if alias == "gelu":
        approx = kwargs.get("approximate", "none")
        return None, f"nn.GELU(approximate={repr(approx)})"

    if alias == "silu":
        return None, "nn.SiLU()"

    if alias == "sigmoid":
        return None, "nn.Sigmoid()"

    if alias == "tanh":
        return None, "nn.Tanh()"

    if alias == "dropout":
        p = kwargs.get("p", 0.5)
        return None, f"nn.Dropout(p={p})"

    if alias == "embedding":
        num_emb = kwargs.get("num_embeddings")
        emb_dim = kwargs.get("embedding_dim")
        return None, f"nn.Embedding({num_emb}, {emb_dim})"

    if alias == "flatten":
        start_dim = kwargs.get("start_dim", 1)
        end_dim = kwargs.get("end_dim", -1)
        return None, f"nn.Flatten(start_dim={start_dim}, end_dim={end_dim})"

    if alias == "layer_norm":
        norm_shape = kwargs.get("normalized_shape")
        eps = kwargs.get("eps", 1e-5)
        elementwise_affine = kwargs.get("elementwise_affine", True)
        return None, f"nn.LayerNorm({norm_shape}, eps={eps}, elementwise_affine={elementwise_affine})"

    if alias == "rms_norm":
        norm_shape = kwargs.get("normalized_shape")
        eps = kwargs.get("eps", 1e-6)
        elementwise_affine = kwargs.get("elementwise_affine", True)
        return None, f"nn.RMSNorm({norm_shape}, eps={eps}, elementwise_affine={elementwise_affine})"

    mod = spec.module
    if mod is not None and mod.__module__.startswith("torch.nn"):
        cls_name = f"nn.{mod.__name__}"
        kwargs_str = ", ".join(f"{k}={_format_value(v)}" for k, v in kwargs.items())
        return None, f"{cls_name}({kwargs_str})"

    # Custom operator module
    cls_name = mod.__name__ if mod is not None else spec.alias.title()
    custom_src = None
    if mod is not None:
        lines = inspect.getsource(mod).splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("class "):
                start_idx = i
                break
        custom_src = "\n".join(lines[start_idx:])

    kwargs_str = ", ".join(f"{k}={_format_value(v)}" for k, v in kwargs.items())
    return custom_src, f"{cls_name}({kwargs_str})"


def to_python_source(plan: ResolvedPlan, class_name: str = "Model") -> str:
    """Emit a standalone, self-contained PyTorch module from a ResolvedPlan."""
    registry = plan.registry
    if registry is None:
        from .registry import Registry
        registry = Registry.builtins()

    specs = {node.id: registry.by_identity(node.op) for node in plan.nodes}

    custom_classes: dict[str, str] = {}
    node_calls: dict[str, str] = {}

    for node in plan.nodes:
        spec = specs[node.id]
        kwargs = _get_constructor_kwargs(spec, node)
        custom_src, call_str = _format_module_constructor(spec, node, kwargs)
        if custom_src is not None:
            mod_name = spec.module.__name__
            if mod_name not in custom_classes:
                custom_classes[mod_name] = custom_src
        node_calls[node.id] = call_str

    output_lines = [
        "# Generated by HNDL v0.10.0; standalone PyTorch model",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
    ]

    for cls_src in custom_classes.values():
        output_lines.append(cls_src)
        output_lines.append("")

    output_lines.append(f"class {class_name}(torch.nn.Module):")
    output_lines.append("    def __init__(self):")
    output_lines.append("        super().__init__()")
    output_lines.append("        self.nodes = nn.ModuleDict()")

    for node in plan.nodes:
        call_str = node_calls[node.id]
        output_lines.append(f"        self.nodes['n_{node.id}'] = {call_str}")

    # Weight tying assignments in __init__
    ties_present = False
    for node in plan.nodes:
        if node.source and "tie" in node.source:
            if not ties_present:
                output_lines.append("")
                output_lines.append("        # Declarative weight tying")
                ties_present = True
            for target_p, source_ref in node.source["tie"].items():
                source_node_id, source_p = source_ref.split(".", 1)
                output_lines.append(f"        self.nodes['n_{node.id}'].{target_p} = self.nodes['n_{source_node_id}'].{source_p}")

    output_lines.append("")

    # Forward method
    input_names = list(plan.inputs.keys())
    sig_inputs = ", ".join(input_names)
    output_lines.append(f"    def forward(self, {sig_inputs}):")

    for node in plan.nodes:
        spec = specs[node.id]
        port_order = spec.input_ports_for(node.args)
        arg_exprs = []
        for port in port_order:
            ref = node.inputs[port]
            if ref.startswith("input:"):
                arg_exprs.append(ref.removeprefix("input:"))
            elif ref.startswith("node:"):
                n_id, p_id = ref.removeprefix("node:").split("/", 1)
                arg_exprs.append(f"t_{n_id}_{p_id}")
            else:
                arg_exprs.append(ref)

        call_args = ", ".join(arg_exprs)
        call_expr = f"self.nodes['n_{node.id}']({call_args})"

        out_shapes = [str(list(s)) for s in node.output_shapes.values()]
        shape_comment = f"# [{node.id}] {spec.alias} -> shape {', '.join(out_shapes)}"
        output_lines.append(f"        {shape_comment}")

        if len(node.outputs) == 1:
            out_var = f"t_{node.id}_{node.outputs[0]}"
            output_lines.append(f"        {out_var} = {call_expr}")
        else:
            out_vars = ", ".join(f"t_{node.id}_{p}" for p in node.outputs)
            output_lines.append(f"        {out_vars} = {call_expr}")

    output_lines.append("")

    if len(plan.outputs) == 1:
        out_name = next(iter(plan.outputs.keys()))
        out_ref = plan.outputs[out_name]["ref"]
        n_id, p_id = out_ref.removeprefix("node:").split("/", 1)
        output_lines.append(f"        return t_{n_id}_{p_id}")
    else:
        out_items = []
        for name, entry in plan.outputs.items():
            n_id, p_id = entry["ref"].removeprefix("node:").split("/", 1)
            out_items.append(f"{repr(name)}: t_{n_id}_{p_id}")
        output_lines.append(f"        return {{{', '.join(out_items)}}}")

    output_lines.append("")
    return "\n".join(output_lines)
