import torch
from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, operator


def _relation(s):
    s.share_batch("x", "params", "out")
    s.rank("params", 2)
    s.equal("x", "out")
    x_shape = s.shape("x")
    if x_shape is None:
        x_shape = s.shape("out")
    if x_shape is None:
        return
    rank = len(x_shape)
    if rank not in (3, 4):
        s.error("E_CONSTRAINT", f"adaptive_norm accepts rank-3 [B, T, C] or rank-4 [B, C, H, W], got rank {rank}")
    s.rank("x", rank)
    s.rank("out", rank)
    c_axis = 2 if rank == 3 else 1
    c_val = x_shape[c_axis]
    params_shape = s.shape("params")
    p_val = params_shape[1] if params_shape is not None else None
    if c_val is not None:
        s.axis("params", 1, 2 * c_val)
    elif p_val is not None:
        if p_val % 2:
            s.error("E_CONSTRAINT", f"params dimension {p_val} must be even (2*C)")
        s.axis("x", c_axis, p_val // 2)
        s.axis("out", c_axis, p_val // 2)


def _reference(module):
    """Explicit two-pass mean and variance calculation without fused kernels."""
    eps = module.eps

    def adaptive_norm(x, params):
        if x.ndim == 4:
            c = x.shape[1]
            mean = x.mean(dim=(2, 3), keepdim=True)
            centered = x - mean
            variance = centered.square().mean(dim=(2, 3), keepdim=True)
            normalized = centered * torch.rsqrt(variance + eps)
            delta_gamma, beta = params[:, :c], params[:, c:]
            return (1 + delta_gamma[:, :, None, None]) * normalized + beta[:, :, None, None]
        if x.ndim == 3:
            c = x.shape[2]
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            variance = centered.square().mean(dim=-1, keepdim=True)
            normalized = centered * torch.rsqrt(variance + eps)
            delta_gamma, beta = params[:, :c], params[:, c:]
            return (1 + delta_gamma[:, None, :]) * normalized + beta[:, None, :]
        raise HNDLError("E_CONSTRAINT", f"adaptive_norm accepts rank-3 or rank-4 inputs, got rank {x.ndim}")

    return adaptive_norm


@operator(
    "adaptive_norm",
    summary="Instance-normalize features, then apply a per-example learned scale and bias.",
    shape="x, params -> out",
    relation=_relation,
    shape_text="x[B, C, H, W] or x[B, T, C]; params[B, 2*C]; out matches x",
    args={"eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False, help="Added to the variance for stability.")},
    reference=_reference,
    examples=[
        Example("z1, z2 = split(64)\nlinear(z1)\nfeatures = reshape(32, 4, 4)\nadaptive_norm(features, z2)",
                ("B", 128), ("B", 32, 4, 4), "32 channels need 64 style values; the projection resolves to 512."),
        Example('linear(256, name="mapping")\nw = relu(name="w")\nlinear(w, name="project")\n'
                'features = reshape(64, 4, 4, name="seed")\nstyle = linear(w, name="style", init={"weight": 0, "bias": 0})\n'
                'adaptive_norm(features, style, name="norm")', ("B", 128), ("B", 64, 4, 4),
                "A mapping network fans out into a projection and a zero-initialized style affine."),
        Example("tokens = linear(32)\nstyle = linear(64)\nadaptive_norm(tokens, style)",
                ("B", 16, 16), ("B", 16, 32),
                "Sequence modulation: 32 feature channels require 64 style parameters."),
    ],
    category="normalization",
)
class AdaptiveNorm(nn.Module):
    """Population instance normalization followed by a learned affine that the
    ``params`` tensor supplies per example: ``params = [delta_gamma, beta]``
    with one value of each per channel.

    ```text
    normalized = (x - mean(x, HW)) / sqrt(var(x, HW) + eps)      # rank 4
    normalized = (x - mean(x, C)) / sqrt(var(x, C) + eps)        # rank 3
    out = (1 + delta_gamma) * normalized + beta
    ```

    A zero ``params`` vector yields the normalized features. There are no
    running statistics and no parameters; train and eval behave identically.
    """

    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x, params):
        if x.ndim == 4:
            c = x.shape[1]
            normalized = F.instance_norm(x, eps=self.eps)
            delta_gamma, beta = params[:, :c], params[:, c:]
            return torch.addcmul(beta[:, :, None, None], normalized, 1 + delta_gamma[:, :, None, None])
        if x.ndim == 3:
            c = x.shape[2]
            normalized = F.layer_norm(x, (c,), eps=self.eps)
            delta_gamma, beta = params[:, :c], params[:, c:]
            return torch.addcmul(beta[:, None, :], normalized, 1 + delta_gamma[:, None, :])
        raise HNDLError("E_CONSTRAINT", f"adaptive_norm accepts rank-3 or rank-4 inputs, got rank {x.ndim}")

    def extra_repr(self):
        return f"eps={self.eps}"
