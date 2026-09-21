import torch
from torch import nn

from ..operator import Arg, Example, operator


def _reference(module):
    def dropout(x):
        return torch.nn.functional.dropout(x, module.p, module.training, inplace=False)
    return dropout


@operator(
    "dropout",
    summary="Randomly zero elements during training and rescale the rest.",
    shape="x[B, ...] -> out[B, ...]",
    args={"p": Arg(float, 0.1, min=0, max=1, exclusive_max=True,
                   help="Probability that an individual element is zeroed while training; 0 disables dropout.")},
    reference=_reference,
    examples=[
        Example("linear(64)\nrelu()\ndropout(0.0)\nlinear()", ("B", 32), ("B", 10),
                "Regularizes the hidden features. The examples pin p=0.0 so that they are reproducible; "
                "use dropout(0.1) in a real network."),
        Example("conv(8, kernel_size=3, padding=1)\ndropout(0.0)\nrelu()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Elementwise on images: every element of [B, C, H, W] is dropped independently."),
        Example("linear(16)\ndropout(0.0)\nlinear()", ("B", 4, 8), ("B", 4, 3),
                "On a [B, T, D] sequence each position and feature is dropped independently."),
    ],
    category="activation",
)
class Dropout(nn.Dropout):
    """Elementwise dropout, computed out of place so shared branches are never
    mutated.

    In **training mode** each element of the input is independently zeroed with
    probability ``p``, and the surviving elements are scaled by ``1 / (1 - p)``
    so the expected value of the activation is unchanged:

    ```text
    m ~ Bernoulli(1 - p)          (drawn per element)
    out = x * m / (1 - p)
    ```

    In **eval mode** (``model.eval()``) dropout is the identity: ``out = x``,
    with no mask and no rescaling. ``p = 0.0`` is the identity in both modes.

    The mask is elementwise and shape agnostic: the operator accepts ``[B, F]``,
    ``[B, T, D]`` and ``[B, C, H, W]`` and drops every element of the tensor
    independently — it does not drop whole channels or whole positions (that
    would be a separate ``dropout2d``-style operator). The output shape always
    equals the input shape, and the operator holds no parameters or buffers.

    ### Randomness

    Dropout is the one deliberate exception to HNDL's "no hidden forward
    randomness" rule: in training mode ``forward`` draws from torch's **global**
    RNG, so two calls on the same input return different results unless the RNG
    is reseeded between them, and a plan containing ``dropout(p > 0)`` is only
    reproducible if you control ``torch.manual_seed`` yourself. Everything the
    resolver sees stays deterministic — the drawn mask never affects shapes,
    arguments, or the plan digest.

    Because of that, the declared examples all use ``p=0.0``: the generic
    example harness builds in training mode and compares two forward passes
    without reseeding, which any ``p > 0`` would fail by construction. The
    operator's own tests cover the statistics and scaling for ``p > 0``.

    The mask is drawn and applied in the plan's compute dtype (float32,
    float16, or bfloat16); nothing is upcast.
    """

    def __init__(self, p):
        super().__init__(p, inplace=False)
