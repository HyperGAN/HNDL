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
            weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            weights = torch.softmax(values, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, indices, weights)
        expert_outputs = torch.stack([expert_forward(e, flat) for e in module.experts], dim=-1)
        return (expert_outputs * gates.unsqueeze(1)).sum(dim=-1).reshape(x.shape)

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
        "top_k": Arg(int, 2, min=1, positional=False,
                     help="Experts each token is routed to; must not exceed experts."),
        "activation": Arg(str, "gelu", positional=False, choices=tuple(ACTIVATIONS),
                          help='Nonlinearity inside each expert: "gelu", "gelu_tanh", "relu" or "silu".'),
        "expert_kind": Arg(str, "mlp", choices=("mlp", "swiglu"), positional=False, since="0.9.0",
                           help='Expert architecture: "mlp" (2-layer) or "swiglu" (3-projection gated).'),
        "bias": Arg(bool, True, positional=False, since="0.9.0",
                    help="Enable learned biases in expert projections."),
        "scoring": Arg(str, "softmax", choices=("softmax", "sigmoid"), positional=False, since="0.9.0",
                       help='Routing gate affinity normalization: "softmax" or "sigmoid".'),
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
                 expert_kind="mlp", bias=True, scoring="softmax"):
        super().__init__()
        self.d_model = D
        self.hidden = hidden
        self.top_k = top_k
        self.activation = activation
        self.expert_kind = expert_kind
        self.bias = bool(bias)
        self.scoring = scoring
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
            weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            weights = torch.softmax(values, dim=-1)

        if torch.compiler.is_compiling():
            gates = torch.zeros(flat.shape[0], len(self.experts),
                                device=flat.device, dtype=weights.dtype).scatter(1, indices, weights)
            out = sum(expert(flat) * gates[:, i : i + 1] for i, expert in enumerate(self.experts))
            return out.reshape(x.shape)

        out = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            rows, cols = torch.where(indices == index)
            if rows.numel() == 0:
                continue
            gate = weights[rows, cols].unsqueeze(-1)
            out.index_add_(0, rows, expert(flat.index_select(0, rows)) * gate)
        return out.reshape(x.shape)

    def extra_repr(self):
        opts = f", expert_kind={self.expert_kind!r}" if self.expert_kind != "mlp" else ""
        if not self.bias:
            opts += ", bias=False"
        if self.scoring != "softmax":
            opts += f", scoring={self.scoring!r}"
        return (f"experts={len(self.experts)}, hidden={self.hidden}, top_k={self.top_k}, "
                f"activation={self.activation!r}{opts}")
