from torch import nn

from ..operator import Example, operator


def _reference(module):
    return lambda x: x.mean(dim=(2, 3))


@operator(
    "global_avg_pool",
    summary="Average each channel over height and width, producing one value per channel.",
    shape="x[B, C, H, W] -> out[B, C]",
    reference=_reference,
    examples=[
        Example("conv(16, kernel_size=3, padding=1)\nrelu()\nglobal_avg_pool()\nlinear()",
                ("B", 3, 16, 16), ("B", 10),
                "A convolutional trunk followed by a global-pooled classifier head; "
                "the head's in_features resolves to the 16 channels."),
        Example("global_avg_pool()", ("B", 32, 7, 7), ("B", 32),
                "Any spatial extent collapses to one value per channel."),
        Example("conv(8, kernel_size=4, stride=2, padding=1)\ngroup_norm(4)\nrelu()\nglobal_avg_pool()\nlinear()",
                ("B", 3, 32, 32), ("B", 4),
                "Spatial size no longer constrains the head, so the trunk can downsample freely."),
    ],
    category="spatial",
)
class GlobalAvgPool2d(nn.Module):
    """Computes ``out[b, c] = mean(x[b, c, :, :])`` on a ``[B, C, H, W]``
    input, returning ``[B, C]``: the spatial axes are reduced away rather
    than kept as size-1 axes, so the result feeds `linear` directly.

    This is `adaptive_avg_pool(1)` followed by `flatten`, written as one
    operation because the pattern is the standard head of a convolutional
    classifier. There are no parameters and train and eval behave
    identically.

    Only rank-4 inputs are accepted, which keeps the axis convention
    unambiguous; reduce a ``[B, C, L]`` sequence with a `reshape` to
    ``[B, C, L, 1]`` first, or use `adaptive_avg_pool`.

    Shape inference: ``C`` flows in both directions, while ``H`` and ``W``
    are *not* inferable backward — every spatial extent produces the same
    output — so they must come from the network input or the preceding
    operation. The mean is accumulated in the input dtype.
    """

    def forward(self, x):
        return x.mean(dim=(2, 3))
