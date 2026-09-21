import torch
from torch import nn

from ..operator import Example, operator


def _reference(module):
    return torch.sigmoid


@operator(
    "sigmoid",
    summary="Logistic sigmoid, squashing values into (0, 1).",
    shape="x[B, ...] -> out[B, ...]",
    examples=[
        Example("linear(1)\nsigmoid()", ("B", 32), ("B", 1),
                "A binary classification head emitting a probability."),
        Example("linear(32)\nsigmoid()\nlinear()", ("B", 6, 16), ("B", 6, 8),
                "On a [B, T, D] sequence the activation applies elementwise at every position."),
        Example("conv(3, kernel_size=3, padding=1)\nsigmoid()", ("B", 8, 16, 16), ("B", 3, 16, 16),
                "A final activation for image generators that emit values in [0, 1]."),
    ],
    reference=_reference,
    category="activation",
)
class Sigmoid(nn.Sigmoid):
    """Elementwise ``out = 1 / (1 + exp(-x))``.

    The output lies strictly in ``(0, 1)``, which makes the operator a natural
    final activation for probabilities and for images normalized to the unit
    interval. Gradients vanish for inputs far from zero, so it is a poor
    choice for hidden layers; prefer `silu` or `gelu` there. When the loss is
    a binary cross entropy, keep the logits and use a fused loss rather than
    stacking `sigmoid` in front of it.

    The operator is elementwise and shape preserving on ``[B, F]``,
    ``[B, T, D]`` and ``[B, C, H, W]`` tensors, has no parameters, behaves
    identically in train and eval mode, and is computed out of place in the
    plan's compute dtype.
    """
