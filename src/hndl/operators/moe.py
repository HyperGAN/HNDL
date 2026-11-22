import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ._activations import ACTIVATION_NAMES
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator
from ._relations import non_image
from ._packing import _global_call_hooks, _has_call_hooks, clear_packed_cache, expert_parameters, guard_expert_input
from .feed_forward import ACTIVATIONS



class _RoutedLinear(torch.autograd.Function):
    """Selected-expert matmuls with a bounded temporary in both passes.

    Backward gathers weights again instead of saving one matrix per route.
    That keeps activation memory proportional to routes, not routes*weights.
    """

    @staticmethod
    def forward(ctx, inputs, weights, expert_ids, bias):
        if torch.is_autocast_enabled(inputs.device.type):
            compute_dtype = torch.get_autocast_dtype(inputs.device.type)
            if inputs.dtype != compute_dtype:
                inputs = inputs.to(compute_dtype)
            if weights.dtype != compute_dtype:
                weights = weights.to(compute_dtype)
            if bias is not None and bias.dtype != compute_dtype:
                bias = bias.to(compute_dtype)
        else:
            if weights.dtype != inputs.dtype:
                weights = weights.to(inputs.dtype)
            if bias is not None and bias.dtype != inputs.dtype:
                bias = bias.to(inputs.dtype)
        ctx.save_for_backward(inputs, weights, expert_ids)
        ctx.has_bias = bias is not None
        weight_bytes = max(1, weights.shape[1] * weights.shape[2] * weights.element_size())
        tile = max(1, min(64, (16 * 1024 * 1024) // weight_bytes))
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
        if gradient.dtype != inputs.dtype:
            gradient = gradient.to(inputs.dtype)
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


def _align_dtype(tensor, dtype):
    return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


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
        "activation": Arg(str, "gelu", positional=False, choices=ACTIVATION_NAMES,
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

    def _can_batch_experts(self):
        if _global_call_hooks():
            return False
        expected = SwiGLUExpert if self.expert_kind == "swiglu" else Expert
        for expert in self.experts:
            if type(expert) is not expected or "forward" in expert.__dict__ or _has_call_hooks(expert):
                return False
            if expected is Expert and expert.activation is not ACTIVATIONS[self.activation]:
                return False
            for projection in expert._modules.values():
                if (type(projection) is not nn.Linear or "forward" in projection.__dict__
                        or _has_call_hooks(projection)):
                    return False
        return True

    def forward(self, x):
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        values, indices = torch.topk(logits, self.top_k, dim=-1)
        if self.scoring == "sigmoid":
            scores = torch.sigmoid(values)
            weights = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        else:
            weights = torch.softmax(values, dim=-1)
        autocast_enabled = torch.is_autocast_enabled(flat.device.type)
        compute_dtype = torch.get_autocast_dtype(flat.device.type) if autocast_enabled else flat.dtype
        if flat.shape[0] == 0:
            return (flat.to(compute_dtype) if flat.dtype != compute_dtype else flat).reshape(x.shape)

        num_experts = len(self.experts)
        element_size = compute_dtype.itemsize
        aligned = (self.d_model * element_size) % 16 == 0 and (self.hidden * element_size) % 16 == 0
        compiling = torch.compiler.is_compiling()
        grouped = (
            self._dispatch == "auto"
            and getattr(self, "_hndl_grouped_mm", True)
            and hasattr(torch, "_grouped_mm")
            and aligned
            and compute_dtype in (torch.bfloat16, torch.float16, torch.float32)
            and (not compiling or compute_dtype == torch.bfloat16)
        )
        estimated_padding = num_experts * flat.shape[0] * (2 * self.d_model + 3 * self.hidden) * element_size
        padded = not grouped and (
            self._dispatch == "padded"
            or (self._dispatch == "auto" and estimated_padding <= self._max_padding_bytes)
        )

        if not self._can_batch_experts():
            expert_ids = indices.reshape(-1)
            route_weights = weights.reshape(-1, 1)
            has_global_hooks = bool(_global_call_hooks())
            out = None
            for expert_id, expert in enumerate(self.experts):
                routes = torch.where(expert_ids == expert_id)[0]
                if (routes.numel() == 0 and not has_global_hooks
                        and not any(_has_call_hooks(m) for m in expert.modules())):
                    continue
                tokens = torch.div(routes, self.top_k, rounding_mode="floor")
                contribution = expert(flat[tokens])
                contribution = contribution * route_weights[routes].to(contribution.dtype)
                if out is None:
                    out = flat.new_zeros(flat.shape, dtype=contribution.dtype)
                elif contribution.dtype != out.dtype:
                    contribution = contribution.to(out.dtype)
                out = out.index_add(0, tokens, contribution)
            if out is None:
                out = flat.new_zeros(flat.shape, dtype=compute_dtype)
            if self.routed_scaling_factor != 1.0:
                out = out * self.routed_scaling_factor
            return out.reshape(x.shape)

        up_w = expert_parameters(self, "up")
        down_w = expert_parameters(self, "down")
        up_b = expert_parameters(self, "up", "bias") if self.bias else None
        down_b = expert_parameters(self, "down", "bias") if self.bias else None
        if self.expert_kind == "swiglu":
            gate_w = expert_parameters(self, "gate")
            gate_b = expert_parameters(self, "gate", "bias") if self.bias else None
        else:
            gate_w = gate_b = None

        if grouped or padded or not compiling:
            expert_ids, order = torch.sort(indices.reshape(-1))
            token_ids = torch.div(order, self.top_k, rounding_mode="floor")
            routed_inputs = flat[token_ids]
            if routed_inputs.dtype != compute_dtype:
                routed_inputs = routed_inputs.to(compute_dtype)
            routing_weights = weights.reshape(-1, 1)[order]
            up_w_c, up_b_c = _align_dtype(up_w, compute_dtype), _align_dtype(up_b, compute_dtype)
            down_w_c, down_b_c = _align_dtype(down_w, compute_dtype), _align_dtype(down_b, compute_dtype)
            gate_w_c, gate_b_c = _align_dtype(gate_w, compute_dtype), _align_dtype(gate_b, compute_dtype)

            if grouped:
                experts = torch.arange(num_experts, device=expert_ids.device)
                offsets = torch.searchsorted(expert_ids, experts, right=True, out_int32=True)
                if self.expert_kind == "swiglu":
                    if gate_w_c.numel() <= 1_048_576:
                        routed_gu = guard_expert_input(
                            self, "gate", guard_expert_input(self, "up", routed_inputs, up_w), gate_w
                        )
                        w_gu = torch.cat((gate_w_c, up_w_c), dim=1)
                        gu = torch._grouped_mm(routed_gu, w_gu.transpose(1, 2), offsets)
                        if self.bias:
                            gu = gu + torch.cat((gate_b_c, up_b_c), dim=1)[expert_ids]
                        gate, up = gu.chunk(2, dim=-1)
                    else:
                        up_in = guard_expert_input(self, "up", routed_inputs, up_w)
                        up = torch._grouped_mm(up_in, up_w_c.transpose(1, 2), offsets)
                        if up_b_c is not None:
                            up = up + up_b_c[expert_ids]
                        gate_in = guard_expert_input(self, "gate", routed_inputs, gate_w)
                        gate = torch._grouped_mm(gate_in, gate_w_c.transpose(1, 2), offsets)
                        if gate_b_c is not None:
                            gate = gate + gate_b_c[expert_ids]
                    hidden = F.silu(gate) * up
                else:
                    up_in = guard_expert_input(self, "up", routed_inputs, up_w)
                    up = torch._grouped_mm(up_in, up_w_c.transpose(1, 2), offsets)
                    if up_b_c is not None:
                        up = up + up_b_c[expert_ids]
                    hidden = self.experts[0].activation(up)
                down_in = guard_expert_input(self, "down", hidden, down_w)
                expert_out = torch._grouped_mm(down_in, down_w_c.transpose(1, 2), offsets)
                if down_b_c is not None:
                    expert_out = expert_out + down_b_c[expert_ids]
                expert_out = expert_out * routing_weights.to(expert_out.dtype)
            elif padded:
                experts = torch.arange(num_experts, device=expert_ids.device)
                offsets = torch.searchsorted(expert_ids, experts)
                rows = torch.arange(expert_ids.shape[0], device=flat.device) - offsets[expert_ids]
                if flat.device.type == "cpu" and not compiling:
                    ends = torch.searchsorted(expert_ids, experts, right=True)
                    max_rows = int((ends - offsets).max().item())
                else:
                    max_rows = flat.shape[0]
                packed_x = routed_inputs.new_zeros((num_experts, max_rows, self.d_model))
                packed_x[expert_ids, rows] = routed_inputs
                up_in = guard_expert_input(self, "up", packed_x, up_w)
                up = torch.bmm(up_in, up_w_c.transpose(1, 2))
                if up_b_c is not None:
                    up = up + up_b_c.unsqueeze(1)
                if self.expert_kind == "swiglu":
                    gate_in = guard_expert_input(self, "gate", packed_x, gate_w)
                    gate = torch.bmm(gate_in, gate_w_c.transpose(1, 2))
                    if gate_b_c is not None:
                        gate = gate + gate_b_c.unsqueeze(1)
                    hidden = F.silu(gate) * up
                else:
                    hidden = self.experts[0].activation(up)
                down_in = guard_expert_input(self, "down", hidden, down_w)
                down = torch.bmm(down_in, down_w_c.transpose(1, 2))
                if down_b_c is not None:
                    down = down + down_b_c.unsqueeze(1)
                expert_out = down[expert_ids, rows] * routing_weights.to(down.dtype)
            else:
                # Sparse bounded-memory dispatch: one contiguous slice per active expert (no weight copies).
                counts = torch.bincount(expert_ids, minlength=num_experts).tolist()
                routed_up = guard_expert_input(self, "up", routed_inputs, up_w)
                up_chunks = routed_up.split(counts, dim=0)
                hidden_parts = []
                if self.expert_kind == "swiglu":
                    routed_gate = guard_expert_input(self, "gate", routed_inputs, gate_w)
                    gate_chunks = routed_gate.split(counts, dim=0)
                    for e, count in enumerate(counts):
                        if count == 0:
                            continue
                        u = F.linear(up_chunks[e], up_w_c[e], None if up_b_c is None else up_b_c[e])
                        g = F.linear(gate_chunks[e], gate_w_c[e], None if gate_b_c is None else gate_b_c[e])
                        hidden_parts.append(F.silu(g) * u)
                else:
                    act = self.experts[0].activation
                    for e, count in enumerate(counts):
                        if count == 0:
                            continue
                        u = F.linear(up_chunks[e], up_w_c[e], None if up_b_c is None else up_b_c[e])
                        hidden_parts.append(act(u))
                hidden = guard_expert_input(self, "down", torch.cat(hidden_parts, dim=0), down_w)
                down_chunks = hidden.split(counts, dim=0)
                out_parts = []
                for e, count in enumerate(counts):
                    if count == 0:
                        continue
                    out_parts.append(
                        F.linear(down_chunks[e], down_w_c[e], None if down_b_c is None else down_b_c[e])
                    )
                expert_out = torch.cat(out_parts, dim=0)
                expert_out = expert_out * routing_weights.to(expert_out.dtype)
        else:
            expert_ids = indices.reshape(-1)
            token_ids = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(self.top_k)
            routed_inputs = flat[token_ids]
            if routed_inputs.dtype != compute_dtype:
                routed_inputs = routed_inputs.to(compute_dtype)
            routing_weights = weights.reshape(-1, 1)
            up = _RoutedLinear.apply(guard_expert_input(self, "up", routed_inputs, up_w), up_w, expert_ids, up_b)
            if self.expert_kind == "swiglu":
                gate = _RoutedLinear.apply(
                    guard_expert_input(self, "gate", routed_inputs, gate_w), gate_w, expert_ids, gate_b
                )
                hidden = F.silu(gate) * up
            else:
                hidden = self.experts[0].activation(up)
            expert_out = _RoutedLinear.apply(
                guard_expert_input(self, "down", hidden, down_w), down_w, expert_ids, down_b
            )
            expert_out = expert_out * routing_weights.to(expert_out.dtype)

        out = flat.new_zeros(flat.shape, dtype=expert_out.dtype).index_add(0, token_ids, expert_out)
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
