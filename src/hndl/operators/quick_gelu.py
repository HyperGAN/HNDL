import torch
from torch import nn

from ..operator import Example, operator

COEFFICIENT = 1.702


def _reference(module):
    def quick_gelu(x):
        return x / (1.0 + torch.exp(-COEFFICIENT * x))

    return quick_gelu


@operator(
    "quick_gelu",
    summary="CLIP's fast GELU approximation, x * sigmoid(1.702 * x).",
    shape="x[B, ...] -> out[B, ...]",
    examples=[
        Example("linear(64)\nquick_gelu()\nlinear()", ("B", 128), ("B", 10),
                "A drop-in replacement for gelu() on [B, F] features."),
        Example("linear(32)\nquick_gelu()\nlinear()", ("B", 6, 16), ("B", 6, 8),
                "On a [B, T, D] sequence the activation applies elementwise at every position."),
        Example("conv(8, kernel_size=3, padding=1)\nquick_gelu()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Elementwise over every channel and spatial position of a [B, C, H, W] tensor."),
    ],
    reference=_reference,
    category="activation",
)
class QuickGELU(nn.Module):
    """Elementwise ``out = x * sigmoid(1.702 * x)``.

    This is the "quick GELU" used by OpenAI's CLIP and by the models derived
    from it. The logistic curve approximates the Gaussian cumulative
    distribution function, so the result tracks `gelu` within about 1e-2 in
    absolute value while costing one sigmoid instead of an erf. A model
    trained with this activation must keep it: it is close to, but not
    numerically interchangeable with, `gelu`.

    The operator is elementwise and shape preserving on ``[B, F]``,
    ``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters and behaves
    identically in train and eval mode. It is computed out of place in the
    input dtype; the 1.702 coefficient is applied as a plain multiply, so
    float16 and bfloat16 activations are never silently upcast.
    """

    def forward(self, x):
        return x * torch.sigmoid(COEFFICIENT * x)
