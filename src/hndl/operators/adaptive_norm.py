import torch
from torch import nn

from ..operator import Arg, Example, operator


@operator(
    "adaptive_norm",
    summary="Instance-normalize features, then apply a per-example learned scale and bias.",
    shape="x[B, C, H, W], params[B, 2*C] -> out[B, C, H, W]",
    args={"eps": Arg(float, 1e-5, min=0, exclusive_min=True, help="Added to the variance for stability.")},
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
        mean = x.mean(dim=(2, 3), keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=(2, 3), keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        delta_gamma, beta = params.split(x.shape[1], dim=1)
        return (1 + delta_gamma[:, :, None, None]) * normalized + beta[:, :, None, None]

    def extra_repr(self):
        return f"eps={self.eps}"
