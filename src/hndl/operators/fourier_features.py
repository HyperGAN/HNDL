import torch
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


@operator(
    "fourier_features",
    summary="Encode coordinates with a fixed random Fourier frequency table.",
    shape="x[B, ..., D] -> out[B, ..., 2*F]",
    args={
        "num_frequencies": Arg(int, dim="F", min=1, max=MAX_DIMENSION_LITERAL // 2,
                               help="Number of frequencies; output width is twice this value."),
        "scale": Arg(float, 8.0, min=0, exclusive_min=True, positional=False,
                     help="Standard deviation of the fixed normal frequency table."),
    },
    examples=[Example('fourier_features(8, scale=8.0)', ("B", 16, 2), ("B", 16, 16),
                      "Encode 2D coordinates with eight sine and eight cosine features.")],
    category="sequence",
)
class FourierFeatures(nn.Module):
    """With fixed ``frequencies ~ Normal(0, scale)``, compute
    ``p = x @ frequencies.T`` and ``concat(sin(p), cos(p), dim=-1)``.
    No extra 2*pi factor is applied. Frequencies have shape [F,D] and are
    a persistent buffer, not trainable parameters. Initialization uses the
    construction RNG; forward draws nothing. To share one table between
    two sets of coordinates, concatenate them before this operator and
    split their features afterwards. Gradients flow to the coordinates.
    """

    def __init__(self, num_frequencies, scale, *, D):
        super().__init__()
        self.register_buffer("frequencies", torch.randn(num_frequencies, D) * scale)

    def forward(self, x):
        p = x @ self.frequencies.t()
        return torch.cat((p.sin(), p.cos()), dim=-1)
