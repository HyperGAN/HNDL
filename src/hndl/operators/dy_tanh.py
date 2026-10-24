"""Dynamic Tanh (dy_tanh) normalization / activation operator."""

import torch
from torch import nn

from ..operator import Arg, Example, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is None:
        rank = 3
    else:
        rank = len(shape)
    s.rank("x", rank)
    s.rank("out", rank)
    s.equal("x", "out")
    dim = s.args.get("dim", 0)
    shape = s.shape("x") or s.shape("out")
    if shape is not None and shape[-1] is not None:
        d = shape[-1]
        if dim == 0:
            s.args["dim"] = d
        elif dim != d:
            s.error("E_CONSTRAINT", f"dy_tanh dim={dim} does not match tensor dim {d}")


def _reference(module):
    alpha = module.alpha
    affine = module.affine
    gamma = module.gamma if affine else None
    beta = module.beta if affine else None

    def run(x):
        scaled = torch.tanh(alpha * x)
        if affine:
            return scaled * gamma + beta
        return scaled

    return run


@operator(
    "dy_tanh",
    identity="dynamic_tanh",
    summary="Dynamic Tanh activation with learned scalar alpha and affine scaling.",
    shape="x[B, ..., D] -> out[B, ..., D]",
    relation=_relation,
    args={
        "dim": Arg(int, inferable=True, dim="D", min=1, help="Feature dimension D."),
        "init_alpha": Arg(float, 1.0, positional=False, help="Initial value for learned scalar alpha."),
        "affine": Arg(bool, True, positional=False, help="Learn affine per-channel gamma and beta."),
    },
    examples=[
        Example(
            "dy_tanh(dim=32)",
            ("B", 8, 32),
            ("B", 8, 32),
            "Dynamic Tanh normalization with learned alpha and affine gamma/beta.",
        ),
    ],
    reference=_reference,
    category="activation",
)
class DyTanh(nn.Module):
    """Dynamic Tanh activation layer: y = gamma * tanh(alpha * x) + beta."""

    def __init__(self, dim=0, init_alpha=1.0, affine=True, *, D=0):
        super().__init__()
        actual_dim = int(dim) if dim > 0 else int(D)
        self.dim = actual_dim
        self.affine = bool(affine)

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        if self.affine and self.dim > 0:
            self.gamma = nn.Parameter(torch.ones(self.dim))
            self.beta = nn.Parameter(torch.zeros(self.dim))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        scaled = torch.tanh(self.alpha * x)
        if self.affine and self.gamma is not None:
            return scaled * self.gamma + self.beta
        return scaled

    def extra_repr(self):
        return f"dim={self.dim}, affine={self.affine}"
