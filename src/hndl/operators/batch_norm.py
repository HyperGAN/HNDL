import torch.nn.functional as F
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator

# The torch module each supported input rank corresponds to: rank 2 [B, C] and
# rank 3 [B, C, L] are BatchNorm1d, rank 4 [B, C, H, W] is BatchNorm2d.
EQUIVALENT = {2: "BatchNorm1d", 3: "BatchNorm1d", 4: "BatchNorm2d"}


def _reference(module):
    def run(x):
        mean = None if module.running_mean is None else module.running_mean.clone()
        variance = None if module.running_var is None else module.running_var.clone()
        training = module.training or not module.track_running_stats
        return F.batch_norm(x, mean, variance, module.weight, module.bias, training,
                            module.momentum, module.eps)
    return run


@operator(
    "batch_norm",
    summary="Normalize each channel over the batch and spatial axes, tracking running statistics.",
    shape="x[B, C, ...] -> out[B, C, ...]",
    args={
        "eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False, help="Added to the variance before the square root."),
        "momentum": Arg(float, 0.1, min=0, max=1, positional=False,
                        help="Weight of the current batch in the running-statistics update."),
        "affine": Arg(bool, True, positional=False, help="Learn a per-channel scale and bias."),
        "track_running_stats": Arg(bool, True, positional=False,
                                   help="Keep running_mean/running_var buffers and use them in eval mode."),
        "num_features": Arg(int, inferable=True, dim="C", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                            help="Channels at axis 1. Inferred from the incoming tensor."),
    },
    reference=_reference,
    examples=[
        Example("conv(16, kernel_size=3, padding=1)\nbatch_norm()\nrelu()\nconv(3, kernel_size=3, padding=1)",
                ("B", 3, 8, 8), ("B", 3, 8, 8), "Normalizes the 16 convolution channels over batch, height and width."),
        Example("linear(64)\nbatch_norm()\nrelu()\nlinear()", ("B", 32), ("B", 10),
                "On a [B, C] tensor every feature is a channel."),
        Example("batch_norm()\nflatten()\nlinear()", ("B", 4, 16), ("B", 10),
                "A [B, C, L] input normalizes each of the 4 channels over the batch and the 16 positions."),
        Example("linear()\nbatch_norm(num_features=64, affine=False)\nrelu()\nlinear()", ("B", 32), ("B", 10),
                "Stating the channel count backward fixes the preceding projection at 64."),
    ],
    category="normalization",
)
class BatchNorm(nn.modules.batchnorm._BatchNorm):
    """Normalizes each channel of axis 1 across the batch and every remaining
    axis:

    ```text
    out = weight * (x - mean) / sqrt(var + eps) + bias
    ```

    Supported ranks are 2 `[B, C]`, 3 `[B, C, L]` and 4 `[B, C, H, W]`; the
    rank is fixed at build time from the resolved input shape, so the module
    is exactly `nn.BatchNorm1d` for ranks 2 and 3 and `nn.BatchNorm2d` for
    rank 4, including its state-dict layout.

    In train mode the statistics come from the current batch, and the
    `running_mean` / `running_var` buffers are updated once per forward as
    `running = (1 - momentum) * running + momentum * batch` (the variance uses
    the unbiased batch estimate, the normalization the biased one);
    `num_batches_tracked` counts the updates. In eval mode the buffers are
    used instead, so the layer becomes a fixed affine map. With
    `track_running_stats=False` there are no buffers and batch statistics are
    used in both modes.

    Parameters are `weight` and `bias` when `affine` is true, one value per
    channel. Statistics are accumulated in the plan's compute dtype.
    """

    def __init__(self, num_features, eps, momentum, affine, track_running_stats, *, input_shapes):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats)
        self.rank = len(input_shapes["x"])

    def _check_input_dim(self, input):
        if input.dim() != self.rank:
            raise HNDLError("E_RUNTIME", f"batch_norm was built for rank {self.rank}, got rank {input.dim()}")

    def extra_repr(self):
        return f"{super().extra_repr()}, rank={self.rank} ({EQUIVALENT[self.rank]})"
