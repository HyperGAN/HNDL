"""Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers."""

import types
import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, Port, operator


def _reference(module):
    eps = module.eps
    linear = module.modulation
    gate_enabled = module.gate

    def run(x, cond):
        params = F.linear(F.silu(cond), linear.weight, linear.bias)
        if gate_enabled:
            shift, scale, gate = params.chunk(3, dim=-1)
        else:
            shift, scale = params.chunk(2, dim=-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + eps)
        out = (1.0 + scale.unsqueeze(1)) * normed + shift.unsqueeze(1)
        if gate_enabled:
            gate_out = gate.unsqueeze(1).expand_as(x)
            return out, gate_out
        return out

    return run


@operator(
    "adaln_zero",
    summary="Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.",
    shape="x[B, T, D], cond[B, D_cond] -> out[B, T, D]",
    args={
        "eps": Arg(
            float,
            1e-6,
            min=0,
            exclusive_min=True,
            positional=False,
            help="Epsilon added to the variance in LayerNorm.",
        ),
        "gate": Arg(
            bool,
            True,
            positional=False,
            since="0.10.0",
            help="Whether to output gate tensor.",
        ),
    },
    examples=[
        Example(
            "c = mean(1)\ncond = linear(c, 32)\nnormed, gate = adaln_zero(x, cond)\nmul(normed, gate)",
            ("B", 16, 64),
            ("B", 16, 64),
            "Modulates sequence features with conditioning vector and produces residual gate.",
        ),
    ],
    reference=_reference,
    category="normalization",
)
class AdaLNZero(nn.Module):
    """Rank-3 adaptive LayerNorm-Zero conditioning modulation for Diffusion Transformers.

    Maps a conditioning vector ``cond[B, D_cond]`` through a zero-initialized
    linear layer to produce per-token shift, scale, and optional dimension-wise gate
    parameters:
    ``shift, scale, gate = Linear(SiLU(cond)).chunk(3, dim=-1)`` (when ``gate=True``)
    ``shift, scale = Linear(SiLU(cond)).chunk(2, dim=-1)`` (when ``gate=False``)
    ``normed = LayerNorm(x, affine=False)``
    ``out = (1 + scale) * normed + shift``
    ``gate_out = gate.expand_as(x)``.
    """

    def __init__(self, eps=1e-6, gate=True, *, D, D_cond):
        super().__init__()
        self.eps = float(eps)
        self.gate = bool(gate)
        self.D = int(D)
        self.D_cond = int(D_cond)
        dim_mult = 3 if self.gate else 2
        self.modulation = nn.Linear(self.D_cond, dim_mult * self.D, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, cond):
        params = self.modulation(F.silu(cond))
        if self.gate:
            shift, scale, gate = params.chunk(3, dim=-1)
            normed = F.layer_norm(x, (self.D,), eps=self.eps)
            out = (1.0 + scale.unsqueeze(1)) * normed + shift.unsqueeze(1)
            gate_out = gate.unsqueeze(1).expand_as(x)
            return out, gate_out
        else:
            shift, scale = params.chunk(2, dim=-1)
            normed = F.layer_norm(x, (self.D,), eps=self.eps)
            out = (1.0 + scale.unsqueeze(1)) * normed + shift.unsqueeze(1)
            return out

    def extra_repr(self):
        s = f"D={self.D}, D_cond={self.D_cond}, eps={self.eps}"
        if not self.gate:
            s += ", gate=False"
        return s


def _output_ports_for(self, args):
    if not args.get("gate", True):
        return ("out",)
    return ("out", "gate")


def _returns_tuple(self, args):
    return args.get("gate", True)


def _relation(s):
    if "gate" in s.outputs:
        s.rank("gate", 3)
        x = s.shape("x")
        if x is not None:
            s.axis("gate", 1, x[1])
            s.axis("gate", 2, x[2])


_op = AdaLNZero.__hndl_operator__
_gate_port = Port(name="gate", pattern=None, dtype="compute", variadic=False)
object.__setattr__(_op, "outputs", (_op.outputs[0], _gate_port))
object.__setattr__(_op, "output_ports", ("out", "gate"))
object.__setattr__(_op, "_port_dtypes", {p.name: p.dtype for p in (*_op.inputs, *_op.outputs)})
object.__setattr__(_op, "output_ports_for", types.MethodType(_output_ports_for, _op))
object.__setattr__(_op, "returns_tuple", types.MethodType(_returns_tuple, _op))
object.__setattr__(_op, "relation", _relation)
