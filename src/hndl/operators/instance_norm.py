import torch.nn.functional as F
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator

# The torch module each supported input rank corresponds to: rank 3 [B, C, L]
# is InstanceNorm1d, rank 4 [B, C, H, W] is InstanceNorm2d.
EQUIVALENT = {3: "InstanceNorm1d", 4: "InstanceNorm2d"}


def _relation(s):
    for port in ("x", "out"):
        shape = s.shape(port)
        if shape is not None and len(shape) < 3:
            s.error("E_CONSTRAINT", "instance_norm needs positions to average over; "
                                    "use [B, C, L] or [B, C, H, W], not a rank-2 tensor")


def _reference(module):
    def run(x):
        return F.instance_norm(x, None, None, module.weight, module.bias, True, module.momentum, module.eps)
    return run


@operator(
    "instance_norm",
    summary="Normalize every channel of every example over its own spatial positions.",
    shape="x[B, C, ...] -> out[B, C, ...]",
    relation=_relation,
    shape_text="x and out share the shape; rank 3 [B, C, L] or rank 4 [B, C, H, W]",
    args={
        "eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False, help="Added to the variance before the square root."),
        "affine": Arg(bool, False, positional=False, help="Learn a per-channel scale and bias."),
        "num_features": Arg(int, inferable=True, dim="C", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                            help="Channels at axis 1. Inferred from the incoming tensor."),
    },
    reference=_reference,
    examples=[
        Example("conv(8, kernel_size=3, padding=1)\ninstance_norm()\nrelu()", ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Each of the 8 channels is normalized per example over height and width."),
        Example("conv(16, kernel_size=3, padding=1)\ninstance_norm(affine=True)\ntanh()\n"
                "conv(3, kernel_size=3, padding=1)", ("B", 3, 16, 16), ("B", 3, 16, 16),
                "The style-transfer arrangement: normalize, then a learned per-channel affine."),
        Example("instance_norm()\nflatten()\nlinear()", ("B", 4, 16), ("B", 10),
                "A [B, C, L] input normalizes each of the 4 channels over its 16 positions."),
    ],
    category="normalization",
)
class InstanceNorm(nn.modules.instancenorm._InstanceNorm):
    """Normalizes each channel of each example independently over its spatial
    positions, with no interaction between examples:

    ```text
    out[b, c] = weight[c] * (x[b, c] - mean(x[b, c])) / sqrt(var(x[b, c]) + eps) + bias[c]
    ```

    Supported ranks are 3 `[B, C, L]` and 4 `[B, C, H, W]`; the rank is fixed
    at build time from the resolved input shape, so the module is exactly
    `nn.InstanceNorm1d` for rank 3 and `nn.InstanceNorm2d` for rank 4. A
    rank-2 `[B, C]` input is rejected during resolution, because a single
    value per channel has no variance to normalize.

    There are no running statistics (`track_running_stats` is always false),
    so train and eval mode behave identically and the layer is deterministic
    per example. Parameters are `weight` and `bias` when `affine` is true, one
    value per channel; with the default `affine=False` the layer has no
    parameters and no buffers at all.

    Statistics are computed in the plan's compute dtype. A constant channel
    normalizes to zero, up to `eps`.
    """

    def __init__(self, num_features, eps, affine, *, input_shapes):
        super().__init__(num_features, eps=eps, affine=affine, track_running_stats=False)
        self.rank = len(input_shapes["x"])

    def _check_input_dim(self, input):
        if input.dim() != self.rank:
            raise HNDLError("E_RUNTIME", f"instance_norm was built for rank {self.rank}, got rank {input.dim()}")

    def _get_no_batch_dim(self):
        return self.rank - 1

    def extra_repr(self):
        return f"{super().extra_repr()}, rank={self.rank} ({EQUIVALENT[self.rank]})"
