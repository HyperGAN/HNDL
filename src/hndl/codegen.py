"""Standalone zero-dependency PyTorch code generation from ResolvedPlan."""

from __future__ import annotations

import keyword
from typing import TYPE_CHECKING

from .errors import HNDLError
from ._source_bundle import SourceBundle
from .types import plain
from ._version import __version__

if TYPE_CHECKING:
    from .types import ResolvedPlan


def _constructor(spec, node, kwargs, bundle, submodels) -> str:
    """Return the PyTorch expression that constructs this node."""
    alias = spec.alias if spec.module.__module__.startswith("hndl.operators.") else ""
    if alias == "linear" and not kwargs.get("equalized") and not kwargs.get("spectral_norm"):
        in_f = kwargs["in_features"]
        out_f = kwargs["out_features"]
        bias = kwargs.get("bias", True)
        return f"nn.Linear({in_f}, {out_f}, bias={bias})"

    if alias == "relu":
        return "nn.ReLU()"

    if alias == "gelu":
        approx = kwargs.get("approximate", "none")
        return f"nn.GELU(approximate={repr(approx)})"

    if alias == "silu":
        return "nn.SiLU()"

    if alias == "sigmoid":
        return "nn.Sigmoid()"

    if alias == "tanh":
        return "nn.Tanh()"

    if alias == "dropout":
        p = kwargs.get("p", 0.5)
        return f"nn.Dropout(p={p})"

    if alias == "embedding":
        num_emb = kwargs.get("num_embeddings")
        emb_dim = kwargs.get("dim") or kwargs.get("embedding_dim")
        return f"nn.Embedding({num_emb}, {emb_dim})"

    if alias == "flatten":
        start_dim = kwargs.get("start_dim", 1)
        end_dim = kwargs.get("end_dim", -1)
        return f"nn.Flatten(start_dim={start_dim}, end_dim={end_dim})"

    if alias == "layer_norm":
        norm_shape = kwargs.get("D") or kwargs.get("normalized_shape") or node.input_shapes.get("x", [None])[-1]
        eps = kwargs.get("eps", 1e-5)
        elementwise_affine = kwargs.get("affine", kwargs.get("elementwise_affine", True))
        bias = kwargs.get("bias", True)
        return f"nn.LayerNorm({norm_shape}, eps={eps}, elementwise_affine={elementwise_affine}, bias={bias})"

    if alias == "rms_norm" and not kwargs.get("zero_centered"):
        norm_shape = kwargs.get("D") or kwargs.get("normalized_shape") or node.input_shapes.get("x", [None])[-1]
        eps = kwargs.get("eps", 1e-6)
        elementwise_affine = kwargs.get("affine", kwargs.get("elementwise_affine", True))
        return f"nn.RMSNorm({norm_shape}, eps={eps}, elementwise_affine={elementwise_affine})"

    inner_plan = getattr(spec.module, "_subgraph_plan", None)
    if inner_plan is not None:
        sub_cls_name = f"Submodel_{spec.alias}"
        if sub_cls_name not in submodels:
            submodels[sub_cls_name] = _model_source(inner_plan, sub_cls_name, bundle, submodels, True)
        return f"{sub_cls_name}()"

    if alias == "pretrained":
        raise ValueError("the model requires its registered checkpoint provider")
    cls_name = bundle.include(spec.module)
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return f"{cls_name}({args})"


def _ref_expr(ref):
    if ref.startswith("input:"):
        return ref.removeprefix("input:")
    node, port = ref.removeprefix("node:").split("/", 1)
    return f"t_{node}_{port}"


def _custom_outputs(result, ports):
    if isinstance(result, dict) and set(result) == set(ports):
        return tuple(result[port] for port in ports)
    if isinstance(result, (tuple, list)) and len(result) == len(ports):
        return tuple(result)
    if len(ports) == 1:
        return (result,)
    raise ValueError(f"Expected values for output ports {ports}")


def _model_source(plan, class_name, bundle, submodels, is_submodule):
    from .registry import Registry
    from .torch import _symbol_values

    registry = plan.registry or Registry.builtins()
    specs = {node.id: registry.by_identity(node.op) for node in plan.nodes}
    node_calls = {}
    for node in plan.nodes:
        spec = specs[node.id]
        kwargs = {name: arg.default for name, arg in spec.args.items() if arg.since is not None}
        kwargs.update(node.args)
        symbols = _symbol_values(spec, node)
        kwargs.update((name, symbols[name]) for name in spec.init_symbols)
        kwargs.update((name, dict(getattr(node, name))) for name in spec.init_shapes)
        try:
            node_calls[node.id] = _constructor(spec, node, kwargs, bundle, submodels)
        except (OSError, TypeError, ValueError, KeyError, SyntaxError) as exc:
            raise HNDLError("E_UNSUPPORTED", f"Cannot emit standalone {spec.alias}: {exc}", node=node.id) from exc

    output_lines = []
    output_lines.append(f"class {class_name}(torch.nn.Module):")
    output_lines.append("    def __init__(self):")
    output_lines.append("        super().__init__()")
    output_lines.append("        self.nodes = nn.ModuleDict()")

    if plan.nodes:
        output_lines.extend([
            "        original_dtype = torch.get_default_dtype()",
            f"        torch.set_default_dtype(torch.{plan.dtype})",
            "        try:",
        ])
        for node in plan.nodes:
            output_lines.append(f"            self.nodes['n_{node.id}'] = {node_calls[node.id]}")
        output_lines.extend([
            "        finally:",
            "            torch.set_default_dtype(original_dtype)",
            f"        self.to(dtype=torch.{plan.dtype})",
        ])

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

    from .torch import _apply_initializer, _prepare_parameter_settings
    for node in plan.nodes:
        if (not node.initialization["overrides"] and not node.trainability["overrides"]
                and node.trainability["default"] is None):
            continue
        prepare = bundle.include(_prepare_parameter_settings)
        apply = bundle.include(_apply_initializer)
        settings = (f"SimpleNamespace(id={node.id!r}, initialization={plain(node.initialization)!r}, "
                    f"trainability={plain(node.trainability)!r})")
        output_lines += [
            f"        settings = {settings}",
            "        with torch.no_grad():",
            f"            for param, name, value, flag in {prepare}(self.nodes['n_{node.id}'], settings, torch.{plan.dtype}):",
            "                if value is not None:",
            f"                    {apply}(param, value, name, settings.id)",
            "                if flag is not None:",
            "                    param.requires_grad_(flag)",
        ]

    # Forward method
    input_names = list(plan.inputs.keys())
    sig_inputs = ", ".join(input_names)
    output_lines.append(f"    def forward(self, {sig_inputs}):")

    for node in plan.nodes:
        spec = specs[node.id]
        port_order = spec.input_ports_for(node.args)
        call_args = ", ".join(_ref_expr(node.inputs[port]) for port in port_order)
        call_expr = f"self.nodes['n_{node.id}']({call_args})"

        out_shapes = [str(list(s)) for s in node.output_shapes.values()]
        shape_comment = f"# [{node.id}] {spec.alias} -> shape {', '.join(out_shapes)}"
        output_lines.append(f"        {shape_comment}")

        out_vars = ", ".join(f"t_{node.id}_{port}" for port in node.outputs)
        if spec.module.__module__.startswith("hndl.operators.") or getattr(spec.module, "_subgraph_plan", None):
            if len(node.outputs) == 1 and spec.returns_tuple(node.args):
                out_vars += ","
            output_lines.append(f"        {out_vars} = {call_expr}")
        else:
            split = bundle.include(_custom_outputs)
            output_lines.append(f"        {out_vars}, = {split}({call_expr}, {tuple(node.outputs)!r})")

    outputs = {name: _ref_expr(entry["ref"]) for name, entry in plan.outputs.items()}
    if len(outputs) == 1:
        result = next(iter(outputs.values()))
    elif is_submodule:
        result = "(" + ", ".join(outputs.values()) + ")"
    else:
        result = "{" + ", ".join(f"{name!r}: {expr}" for name, expr in outputs.items()) + "}"
    output_lines.append(f"        return {result}")

    output_lines.append("")
    return "\n".join(output_lines)


def to_python_source(plan: ResolvedPlan, class_name: str = "Model", *, is_submodule: bool = False) -> str:
    """Emit a standalone, self-contained PyTorch module from a resolved plan."""
    if not class_name.isidentifier() or keyword.iskeyword(class_name):
        raise ValueError("class_name must be a Python identifier")
    bundle = SourceBundle()
    submodels = {}
    model = _model_source(plan, class_name, bundle, submodels, is_submodule)
    return "\n".join([
        f"# Generated by HNDL v{__version__}; standalone PyTorch model",
        "from __future__ import annotations",
        "from types import SimpleNamespace",
        "import torch",
        "from torch import nn",
        "",
        *bundle.render(), *submodels.values(), model,
    ])
