from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    shape = s.shape("x")
    channels = shape[1] if shape is not None else None
    if channels is not None and channels % s.args["num_groups"]:
        s.error("E_CONSTRAINT", f"num_groups={s.args['num_groups']} must divide channels={channels}")


@operator(
    "group_norm",
    summary="Normalize channel groups per example, with learned per-channel affine.",
    shape="x[B, C, ...] -> out[B, C, ...]",
    relation=_relation,
    args={
        "num_groups": Arg(int, min=1, help="Number of channel groups; must divide the channel count."),
        "num_channels": Arg(int, inferable=True, dim="C", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                            help="Channels at axis 1. Inferred from the incoming tensor."),
        "eps": Arg(float, 1e-5, min=0, exclusive_min=True, positional=False, help="Added to the variance for stability."),
        "affine": Arg(bool, True, positional=False, help="Learn per-channel scale and bias."),
    },
    examples=[Example("conv(16, kernel_size=3, padding=1)\ngroup_norm(4)\nrelu()", ("B", 3, 8, 8), ("B", 16, 8, 8))],
    category="normalization",
)
class GroupNorm(nn.GroupNorm):
    """Splits the channel axis into ``num_groups`` groups and normalizes each
    group over its channels and spatial positions using population
    statistics. Behavior is identical in train and eval mode. Parameters are
    ``weight`` and ``bias`` when ``affine`` is true.
    """

    def __init__(self, num_groups, num_channels, eps, affine):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
