import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator
from ._relations import non_image
from .feed_forward import ACTIVATIONS


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
        self.router = nn.Linear(D, experts, bias=False)
        if expert_kind == "swiglu":
            self.experts = nn.ModuleList(SwiGLUExpert(D, hidden, bias=bias) for _ in range(experts))
        else:
            self.experts = nn.ModuleList(Expert(D, hidden, activation, bias=bias) for _ in range(experts))

    def forward(self, x):
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        values, indices = torch.topk(logits, self.top_k, dim=-1)
        if self.scoring == "sigmoid":
            scores = torch.sigmoid(values)
            weights = scores / (scores.sum(dim=-1, keepdim=True).clamp(min=1e-4))
        else:
            weights = torch.softmax(values, dim=-1)

        N = flat.shape[0]
        if N == 0:
            return flat.reshape(x.shape)

        E = len(self.experts)
        token_ids = torch.arange(N, device=flat.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        flat_expert_ids = indices.reshape(-1)
        flat_weights = weights.reshape(-1, 1)

        sort_order = torch.argsort(flat_expert_ids)
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_inputs = flat[sorted_token_ids]

        elem_bytes = sorted_inputs.element_size()
        stride_ok = (
            (self.d_model * elem_bytes) % 16 == 0 and (self.hidden * elem_bytes) % 16 == 0
        )
        use_grouped = (
            hasattr(torch, "_grouped_mm")
            and stride_ok
            and sorted_inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and (
                not torch.compiler.is_compiling()
                or sorted_inputs.dtype == torch.bfloat16
            )
        )

        if use_grouped:
            if torch.compiler.is_compiling():
                counts = F.one_hot(sorted_expert_ids, num_classes=E).sum(dim=0)
            else:
                counts = torch.bincount(sorted_expert_ids, minlength=E)
            offs = torch.cumsum(counts, dim=0, dtype=torch.int32)

            if self.expert_kind == "swiglu":
                gate_w = torch.stack([e.gate.weight for e in self.experts]).transpose(1, 2)
                up_w = torch.stack([e.up.weight for e in self.experts]).transpose(1, 2)
                g = torch._grouped_mm(sorted_inputs, gate_w, offs)
                u = torch._grouped_mm(sorted_inputs, up_w, offs)
                if self.bias:
                    gate_b = torch.stack([e.gate.bias for e in self.experts])
                    up_b = torch.stack([e.up.bias for e in self.experts])
                    g = g + gate_b[sorted_expert_ids]
                    u = u + up_b[sorted_expert_ids]
                act = F.silu(g) * u
                down_w = torch.stack([e.down.weight for e in self.experts]).transpose(1, 2)
                expert_out = torch._grouped_mm(act, down_w, offs)
                if self.bias:
                    down_b = torch.stack([e.down.bias for e in self.experts])
                    expert_out = expert_out + down_b[sorted_expert_ids]
            else:
                up_w = torch.stack([e.up.weight for e in self.experts]).transpose(1, 2)
                h = torch._grouped_mm(sorted_inputs, up_w, offs)
                if self.bias:
                    up_b = torch.stack([e.up.bias for e in self.experts])
                    h = h + up_b[sorted_expert_ids]
                act = self.experts[0].activation(h)
                down_w = torch.stack([e.down.weight for e in self.experts]).transpose(1, 2)
                expert_out = torch._grouped_mm(act, down_w, offs)
                if self.bias:
                    down_b = torch.stack([e.down.bias for e in self.experts])
                    expert_out = expert_out + down_b[sorted_expert_ids]
        else:
            counts = F.one_hot(sorted_expert_ids, num_classes=E).sum(dim=0)
            offs = torch.cat([torch.zeros(1, dtype=torch.long, device=flat.device), torch.cumsum(counts, 0)[:-1]])
            intra_idx = torch.arange(sorted_expert_ids.shape[0], device=flat.device) - offs[sorted_expert_ids]

            C_max = N
            packed_inputs = torch.zeros(E, C_max, self.d_model, device=flat.device, dtype=flat.dtype)
            packed_inputs[sorted_expert_ids, intra_idx] = sorted_inputs

            if self.expert_kind == "swiglu":
                gate_w = torch.stack([e.gate.weight for e in self.experts]).transpose(1, 2)
                up_w = torch.stack([e.up.weight for e in self.experts]).transpose(1, 2)
                g = torch.bmm(packed_inputs, gate_w)
                u = torch.bmm(packed_inputs, up_w)
                if self.bias:
                    gate_b = torch.stack([e.gate.bias for e in self.experts]).unsqueeze(1)
                    up_b = torch.stack([e.up.bias for e in self.experts]).unsqueeze(1)
                    g = g + gate_b
                    u = u + up_b
                act = F.silu(g) * u
                down_w = torch.stack([e.down.weight for e in self.experts]).transpose(1, 2)
                expert_out_bmm = torch.bmm(act, down_w)
                if self.bias:
                    down_b = torch.stack([e.down.bias for e in self.experts]).unsqueeze(1)
                    expert_out_bmm = expert_out_bmm + down_b
            else:
                up_w = torch.stack([e.up.weight for e in self.experts]).transpose(1, 2)
                h = torch.bmm(packed_inputs, up_w)
                if self.bias:
                    up_b = torch.stack([e.up.bias for e in self.experts]).unsqueeze(1)
                    h = h + up_b
                act = self.experts[0].activation(h)
                down_w = torch.stack([e.down.weight for e in self.experts]).transpose(1, 2)
                expert_out_bmm = torch.bmm(act, down_w)
                if self.bias:
                    down_b = torch.stack([e.down.bias for e in self.experts]).unsqueeze(1)
                    expert_out_bmm = expert_out_bmm + down_b

            expert_out = expert_out_bmm[sorted_expert_ids, intra_idx]

        expert_out = expert_out * sorted_weights
        out = torch.zeros_like(flat)
        out.index_add_(0, sorted_token_ids, expert_out)
        if self.routed_scaling_factor != 1.0:
            out = out * self.routed_scaling_factor
        return out.reshape(x.shape)

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
