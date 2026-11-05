import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator
from ._relations import non_image
from ._packing import clear_packed_cache, expert_parameters, guard_expert_input
from .feed_forward import ACTIVATIONS



class _RoutedLinear(torch.autograd.Function):
    """Selected-expert matmuls with a bounded temporary in both passes.

    Backward gathers weights again instead of saving one matrix per route.
    That keeps activation memory proportional to routes, not routes*weights.
    """

    @staticmethod
    def forward(ctx, inputs, weights, expert_ids, bias):
        ctx.save_for_backward(inputs, weights, expert_ids)
        ctx.has_bias = bias is not None
        tile = max(1, min(64, 1048576 // (weights.shape[1] * weights.shape[2])))
        ctx.tile = tile
        out = inputs.new_empty((inputs.shape[0], weights.shape[1]))
        for start in range(0, inputs.shape[0], tile):
            end = start + tile
            chosen = weights[expert_ids[start:end]]
            result = torch.bmm(chosen, inputs[start:end].unsqueeze(-1)).squeeze(-1)
            if bias is not None:
                result = result + bias[expert_ids[start:end]]
            out[start:end] = result
        return out

    @staticmethod
    def backward(ctx, gradient):
        inputs, weights, expert_ids = ctx.saved_tensors
        input_gradient = torch.empty_like(inputs) if ctx.needs_input_grad[0] else None
        weight_gradient = torch.zeros_like(weights) if ctx.needs_input_grad[1] else None
        bias_gradient = (gradient.new_zeros((weights.shape[0], weights.shape[1]))
                         if ctx.has_bias and ctx.needs_input_grad[3] else None)
        for start in range(0, inputs.shape[0], ctx.tile):
            end = start + ctx.tile
            ids = expert_ids[start:end]
            grad = gradient[start:end]
            if input_gradient is not None:
                input_gradient[start:end] = torch.bmm(
                    weights[ids].transpose(1, 2), grad.unsqueeze(-1)).squeeze(-1)
            if weight_gradient is not None:
                products = grad.unsqueeze(-1) * inputs[start:end].unsqueeze(1)
                if torch.is_grad_enabled():
                    weight_gradient = weight_gradient.index_add(0, ids, products)
                else:
                    weight_gradient.index_add_(0, ids, products)
            if bias_gradient is not None:
                if torch.is_grad_enabled():
                    bias_gradient = bias_gradient.index_add(0, ids, grad)
                else:
                    bias_gradient.index_add_(0, ids, grad)
        return input_gradient, weight_gradient, None, bias_gradient


def _validate(args):
    if args["top_k"] > args["experts"]:
        raise HNDLError("E_ARGUMENT", f"top_k={args['top_k']} must be <= experts={args['experts']}")


def _reference(module):
    """Dense reference oracle: evaluate router and all experts without sparse indexing."""
    activation = ACTIVATIONS[module.activation]

    def expert_forward(expert, h):
        if module.expert_kind == "swiglu":
            return F.linear(
                F.silu(F.linear(h, expert.gate.weight, expert.gate.bias)) * F.linear(h, expert.up.weight, expert.up.bias),
                expert.down.weight, expert.down.bias,
            )
        return F.linear(activation(F.linear(h, expert.up.weight, expert.up.bias)), expert.down.weight, expert.down.bias)

    def run(x):
        flat = x.reshape(-1, module.d_model)
        logits = F.linear(flat, module.router.weight)
        values, indices = torch.topk(logits, module.top_k, dim=-1)
        if module.scoring == "sigmoid":
            scores = torch.sigmoid(values)
            weights = scores / (scores.sum(dim=-1, keepdim=True).clamp(min=1e-4))
        else:
            weights = torch.softmax(values, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, indices, weights)
        expert_outputs = torch.stack([expert_forward(e, flat) for e in module.experts], dim=-1)
        scaled = (expert_outputs * gates.unsqueeze(1)).sum(dim=-1)
        factor = getattr(module, "routed_scaling_factor", 1.0)
        if factor != 1.0:
            scaled = scaled * factor
        return scaled.reshape(x.shape)

    return run


class Expert(nn.Module):
    """One feed-forward expert: ``down(activation(up(x)))``."""

    def __init__(self, d_model, hidden, activation, bias=True):
        super().__init__()
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        return self.down(self.activation(self.up(x)))


class SwiGLUExpert(nn.Module):
    """One 3-projection SwiGLU expert: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model, hidden, bias=True):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=bias)
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


@operator(
    "moe",
    summary="Sparse mixture of experts: route every token to its top-k feed-forward experts.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    relation=non_image,
    args={
        "experts": Arg(int, min=1, max=MAX_DIMENSION_LITERAL, help="Number of feed-forward experts."),
        "hidden": Arg(int, min=1, max=MAX_DIMENSION_LITERAL, help="Inner width of every expert's feed-forward."),
        "top_k": Arg(int, 2, min=1, positional=True,
                     help="Experts each token is routed to; must not exceed experts."),
        "activation": Arg(str, "gelu", positional=False, choices=tuple(ACTIVATIONS),
                          help='Nonlinearity inside each expert: "gelu", "gelu_tanh", "relu" or "silu".'),
        "expert_kind": Arg(str, "mlp", choices=("mlp", "swiglu"), positional=False, since="0.9.0",
                           help='Expert architecture: "mlp" (2-layer) or "swiglu" (3-projection gated).'),
        "bias": Arg(bool, True, positional=False, since="0.9.0",
                    help="Enable learned biases in expert projections."),
        "scoring": Arg(str, "softmax", choices=("softmax", "sigmoid"), positional=False, since="0.9.0",
                       help='Routing gate affinity normalization: "softmax" or "sigmoid".'),
        "routed_scaling_factor": Arg(
            float,
            1.0,
            min=0.0,
            exclusive_min=True,
            positional=False,
            since="0.10.0",
            help="Multiplier applied to routed expert outputs.",
        ),
    },
    validate=_validate,
    reference=_reference,
    examples=[
        Example("linear(16)\nmoe(4, 32)\nlinear()", ("B", 8), ("B", 4),
                "Four experts, two of which see each example; the mixture keeps the width at 16."),
        Example("moe(2, 16, top_k=1)\nlinear()", ("B", 4, 8), ("B", 4, 3),
                "On a [B, T, D] sequence every position is routed independently; top_k=1 is hard routing."),
        Example('moe(3, 24, activation="silu")', ("B", 6, 12), ("B", 6, 12),
                "The operator preserves its input shape, so it drops into a residual stack unchanged."),
        Example('moe(4, 32, expert_kind="swiglu", bias=False)', ("B", 8), ("B", 8),
                "Four bias-free SwiGLU experts routed with top-2 softmax gating."),
        Example('moe(4, 32, scoring="sigmoid")', ("B", 8), ("B", 8),
                "Top-2 routing with sigmoid affinity scoring normalized over selected experts."),
    ],
    category="sequence",
)
class MixtureOfExperts(nn.Module):
    """A sparsely gated mixture of experts over the last axis of ``[B, D]`` or
    ``[B, T, D]`` inputs. Every row of the flattened tensor (one example, or
    one position of one example) is a *token* and is routed independently.

    ```text
    logits  = router(x)                        # [N, experts], bias-free linear
    v, idx  = topk(logits, top_k)              # the top_k experts per token
    w       = softmax(v) or sigmoid(v)/sum     # renormalized over the selection
    out     = sum_k w[:, k] * expert[idx[:, k]](x)
    ```

    Each expert is either ``down(activation(up(x)))`` (for ``expert_kind="mlp"``)
    or ``down(silu(gate(x)) * up(x))`` (for ``expert_kind="swiglu"``).
    The experts live in an ``nn.ModuleList`` named ``experts``; the router is
    ``router.weight``.
    """

    def __init__(self, experts, hidden, top_k, activation, *, D,
                 expert_kind="mlp", bias=True, scoring="softmax", routed_scaling_factor=1.0):
        super().__init__()
        self.d_model = D
        self.hidden = hidden
        self.top_k = top_k
        self.activation = activation
        self.expert_kind = expert_kind
        self.bias = bool(bias)
        self.scoring = scoring
        self.routed_scaling_factor = float(routed_scaling_factor)
        self._dispatch = "auto"
        self._max_padding_bytes = 16 * 1024 * 1024
        self.router = nn.Linear(D, experts, bias=False)
        if expert_kind == "swiglu":
            self.experts = nn.ModuleList(SwiGLUExpert(D, hidden, bias=bias) for _ in range(experts))
        else:
            self.experts = nn.ModuleList(Expert(D, hidden, activation, bias=bias) for _ in range(experts))

    def configure_dispatch(self, mode="auto", *, max_padding_bytes=16 * 1024 * 1024):
        """Select automatic, strictly sparse, or padded expert dispatch.

        Automatic dispatch prefers supported grouped GEMM. Its portable
        fallback uses small padded batches below the estimated activation
        budget, and bounded selected-route execution above it.
        """
        if mode not in ("auto", "sparse", "padded"):
            raise ValueError("expert dispatch must be 'auto', 'sparse', or 'padded'")
        if type(max_padding_bytes) is not int or max_padding_bytes < 0:
            raise ValueError("max_padding_bytes must be a nonnegative integer")
        self._dispatch = mode
        self._max_padding_bytes = max_padding_bytes
        return self

    def forward(self, x):
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        values, indices = torch.topk(logits, self.top_k, dim=-1)
        if self.scoring == "sigmoid":
            scores = torch.sigmoid(values)
            weights = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        else:
            weights = torch.softmax(values, dim=-1)
        if flat.shape[0] == 0:
            return flat.reshape(x.shape)

        token_ids = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(self.top_k)
        expert_ids = indices.reshape(-1)
        routed_inputs = flat[token_ids]

        # Grouped GEMM is useful where available. Portable dispatch keeps
        # small dense batches fast, with a bounded-memory sparse alternative.
        element_size = routed_inputs.element_size()
        aligned = (self.d_model * element_size) % 16 == 0 and (self.hidden * element_size) % 16 == 0
        grouped = (self._dispatch == "auto" and getattr(self, "_hndl_grouped_mm", True)
                   and hasattr(torch, "_grouped_mm")
                   and aligned and routed_inputs.dtype in (torch.bfloat16, torch.float16, torch.float32)
                   and (not torch.compiler.is_compiling() or routed_inputs.dtype == torch.bfloat16))
        estimated_padding = len(self.experts) * flat.shape[0] * (2 * self.d_model + 3 * self.hidden) * element_size
        padded = (not grouped and (self._dispatch == "padded"
                  or (self._dispatch == "auto" and estimated_padding <= self._max_padding_bytes)))
        if flat.device.type == "cpu" and not torch.compiler.is_compiling() and not grouped and not padded:
            out = torch.zeros_like(flat)
            route_weights = weights.reshape(-1, 1)
            for expert_id, expert in enumerate(self.experts):
                routes = torch.where(expert_ids == expert_id)[0]
                tokens = token_ids[routes]
                contribution = expert(flat[tokens]) * route_weights[routes]
                out = out.index_add(0, tokens, contribution)
            return (out * self.routed_scaling_factor).reshape(x.shape)

        up_w = expert_parameters(self, "up")
        down_w = expert_parameters(self, "down")
        up_b = expert_parameters(self, "up", "bias") if self.bias else None
        down_b = expert_parameters(self, "down", "bias") if self.bias else None
        if grouped or padded:
            order = torch.argsort(expert_ids)
            routed_inputs = routed_inputs[order]
            expert_ids = expert_ids[order]
            token_ids = token_ids[order]
            routing_weights = weights.reshape(-1, 1)[order]
            counts = F.one_hot(expert_ids, num_classes=len(self.experts)).sum(dim=0)
            if grouped:
                offsets = counts.cumsum(0).to(torch.int32)

                def linear(inputs, weight, bias):
                    result = torch._grouped_mm(inputs, weight.transpose(1, 2), offsets)
                    return result if bias is None else result + bias[expert_ids]
            else:
                offsets = counts.cumsum(0) - counts
                rows = torch.arange(expert_ids.shape[0], device=flat.device) - offsets[expert_ids]

                def linear(inputs, weight, bias):
                    packed = inputs.new_zeros((len(self.experts), flat.shape[0], inputs.shape[-1]))
                    packed[expert_ids, rows] = inputs
                    result = torch.bmm(packed, weight.transpose(1, 2))
                    if bias is not None:
                        result = result + bias.unsqueeze(1)
                    return result[expert_ids, rows]
        else:
            routing_weights = weights.reshape(-1, 1)

            def linear(inputs, weight, bias):
                return _RoutedLinear.apply(inputs, weight, expert_ids, bias)

        up = linear(guard_expert_input(self, "up", routed_inputs, up_w), up_w, up_b)
        if self.expert_kind == "swiglu":
            gate_w = expert_parameters(self, "gate")
            gate_b = expert_parameters(self, "gate", "bias") if self.bias else None
            gate = linear(guard_expert_input(self, "gate", routed_inputs, gate_w), gate_w, gate_b)
            hidden = F.silu(gate) * up
        else:
            hidden = self.experts[0].activation(up)
        expert_out = linear(guard_expert_input(self, "down", hidden, down_w), down_w, down_b) * routing_weights
        out = torch.zeros_like(flat).index_add(0, token_ids, expert_out)
        if self.routed_scaling_factor != 1.0:
            out = out * self.routed_scaling_factor
        return out.reshape(x.shape)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if getattr(self, "_hndl_packing", False):
            clear_packed_cache(self)
        return result

    def extra_repr(self):
        opts = f", expert_kind={self.expert_kind!r}" if self.expert_kind != "mlp" else ""
        if not self.bias:
            opts += ", bias=False"
        if self.scoring != "softmax":
            opts += f", scoring={self.scoring!r}"
        if getattr(self, "routed_scaling_factor", 1.0) != 1.0:
            opts += f", routed_scaling_factor={self.routed_scaling_factor}"
        return (f"experts={len(self.experts)}, hidden={self.hidden}, top_k={self.top_k}, "
                f"activation={self.activation!r}{opts}")
