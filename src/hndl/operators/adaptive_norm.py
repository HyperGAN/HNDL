import torch
from torch import nn
from torch.nn import functional as F

from ..operator import Arg, Example, operator


def _reference(module):
    """Explicit two-pass mean and variance calculation without fused kernels."""
    eps = module.eps

    def adaptive_norm(x, params):
        c = x.shape[1]
        mean = x.mean(dim=(2, 3), keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=(2, 3), keepdim=True)
        normalized = centered * torch.rsqrt(variance + eps)
        delta_gamma, beta = params[:, :c], params[:, c:]
        return (1 + delta_gamma[:, :, None, None]) * normalized + beta[:, :, None, None]

    return adaptive_norm


@operator(
    "adaptive_norm",
    summary="Instance-normalize features, then apply a per-example learned scale and bias.",
    shape="x[B, C, H, W], params[B, 2*C] -> out[B, C, H, W]",
    args={"eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False, help="Added to the variance for stability.")},
    reference=_reference,
    examples=[
        Example("z1, z2 = split(64)\nlinear(z1)\nfeatures = reshape(32, 4, 4)\nadaptive_norm(features, z2)",
                ("B", 128), ("B", 32, 4, 4), "32 channels need 64 style values; the projection resolves to 512."),
        Example('linear(256, name="mapping")\nw = relu(name="w")\nlinear(w, name="project")\n'
                'features = reshape(64, 4, 4, name="seed")\nstyle = linear(w, name="style", init={"weight": 0, "bias": 0})\n'
                'adaptive_norm(features, style, name="norm")', ("B", 128), ("B", 64, 4, 4),
                "A mapping network fans out into a projection and a zero-initialized style affine."),
    ],
    category="normalization",
)
class AdaptiveNorm(nn.Module):
    """Population instance normalization followed by a learned affine that the
    ``params`` tensor supplies per example: ``params = [delta_gamma, beta]``
    with one value of each per channel.

    ```text
    normalized = (x - mean(x, HW)) / sqrt(var(x, HW) + eps)
    out = (1 + delta_gamma)[:, :, None, None] * normalized + beta[:, :, None, None]
    ```

    A zero ``params`` vector yields the normalized features. There are no
    running statistics and no parameters; train and eval behave identically.
    """

    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x, params):
        c = x.shape[1]
        normalized = F.instance_norm(x, eps=self.eps)
        delta_gamma, beta = params[:, :c], params[:, c:]
        return torch.addcmul(beta[:, :, None, None], normalized, 1 + delta_gamma[:, :, None, None])

    def extra_repr(self):
        return f"eps={self.eps}"
