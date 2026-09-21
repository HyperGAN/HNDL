import torch.nn.functional as F
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator

KERNEL = 3
PADDING = 1
NORMS = ("batch_norm", "group_norm", "none")


def _relation(s):
    """Spatial arithmetic of the two 3x3 convolutions (k=3, p=1, dilation=1).

    Only the first convolution carries the stride, so the block maps
    ``H_out = floor((H_in + 2*1 - 3) / stride) + 1``, which is
    ``ceil(H_in / stride)``. The inverse of that floor division is an
    interval of input extents, reported through ``s.interval`` so that a
    stride greater than one stays ambiguous instead of being guessed.
    """
    args = s.args
    stride = args["stride"]
    x, out = s.rank("x", 4), s.rank("out", 4)
    if args["norm"] == "group_norm" and out[1] is not None and out[1] % args["groups"]:
        s.error("E_CONSTRAINT", f"groups={args['groups']} must divide out_channels={out[1]} for group_norm")
    for axis in (2, 3):
        extent, target = x[axis], out[axis]
        if extent is not None:
            s.axis("out", axis, (extent + 2 * PADDING - KERNEL) // stride + 1)
        if target is not None:
            lower = max(1, (target - 1) * stride - 2 * PADDING + KERNEL - 1 + 1)
            upper = target * stride - 2 * PADDING + KERNEL - 1
            if lower > upper:
                s.error("E_CONSTRAINT", "Residual block inverse has no positive input extent")
            s.interval("x", axis, lower, upper)


def _norm(kind, channels, groups):
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels)
    if kind == "group_norm":
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def _reference(module):
    """A functional rebuild of the block from the module's own parameters."""

    def apply(layer, value):
        if isinstance(layer, nn.Conv2d):
            return F.conv2d(value, layer.weight, layer.bias, layer.stride, layer.padding)
        if isinstance(layer, nn.BatchNorm2d):
            # Clone the running statistics so the reference never updates them.
            return F.batch_norm(value, layer.running_mean.clone(), layer.running_var.clone(), layer.weight,
                                layer.bias, layer.training, layer.momentum, layer.eps)
        if isinstance(layer, nn.GroupNorm):
            return F.group_norm(value, layer.num_groups, layer.weight, layer.bias, layer.eps)
        return value

    def run(x):
        out = F.relu(apply(module.norm1, apply(module.conv1, x)))
        out = apply(module.norm2, apply(module.conv2, out))
        identity = x
        for layer in (module.shortcut if isinstance(module.shortcut, nn.Sequential) else (module.shortcut,)):
            identity = apply(layer, identity)
        return F.relu(out + identity)

    return run


@operator(
    "resblock",
    summary="Residual basic block: two 3x3 convolutions with a normalized shortcut.",
    shape="x[B, C_in, H_in, W_in] -> out[B, C_out, H_out, W_out]",
    relation=_relation,
    shape_text="H_out = floor((H_in + 2 - 3) / stride) + 1 = ceil(H_in / stride), same for W",
    args={
        "out_channels": Arg(int, inferable=True, dim="C_out", min=1, max=MAX_DIMENSION_LITERAL,
                            help="Channels the block produces. Omit to infer from the consumer."),
        "in_channels": Arg(int, inferable=True, dim="C_in", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Input channels. Normally inferred from the incoming tensor."),
        "stride": Arg(int, 1, min=1, positional=False,
                      help="Stride of the first convolution and of the shortcut; 2 halves height and width."),
        "norm": Arg(str, "batch_norm", positional=False, choices=NORMS,
                    help='Normalization after each convolution: "batch_norm", "group_norm", or "none".'),
        "groups": Arg(int, 8, min=1, positional=False,
                      help='Channel groups when norm="group_norm"; must divide the channel count. Ignored otherwise.'),
    },
    reference=_reference,
    examples=[
        Example("resblock(16)\nresblock(32, stride=2)\nflatten()\nlinear()", ("B", 3, 8, 8), ("B", 10),
                "Two blocks; the strided one halves height and width before the classifier head."),
        Example('resblock(8, norm="group_norm", groups=4)', ("B", 3, 8, 8), ("B", 8, 8, 8),
                "Group normalization with 4 groups over 8 channels; batch size never affects the statistics."),
        Example('resblock(norm="none")', ("B", 8, 16, 16), ("B", 16, 16, 16),
                "A plain residual convolution block; the width 16 is inferred from the output contract."),
    ],
    category="vision",
)
class ResBlock(nn.Module):
    """The ResNet *basic block* on ``[B, C, H, W]`` images:

    ```text
    y = norm2(conv2(relu(norm1(conv1(x)))))
    out = relu(y + shortcut(x))
    ```

    Both convolutions use a 3x3 kernel with padding 1; only ``conv1`` carries
    the stride, so height and width map as
    ``H_out = floor((H_in + 2 - 3) / stride) + 1``, which equals
    ``ceil(H_in / stride)`` — stride 1 preserves the spatial extent and
    stride 2 halves it (rounding up).

    The shortcut is an ``nn.Identity`` when ``stride == 1`` and the channel
    count is unchanged. Otherwise it is an ``nn.Sequential`` of a 1x1
    convolution with the same stride followed by its own normalization, so
    the residual sum is well defined. Submodules are named ``conv1``,
    ``norm1``, ``conv2``, ``norm2`` and ``shortcut``; parameter targets for
    ``init`` and ``trainable`` use those paths, for example
    ``init={"norm2.weight": 0}`` for a zero-initialized residual branch.

    ``norm="batch_norm"`` keeps running mean and variance buffers, so train
    and eval mode differ: training normalizes with the statistics of the
    current batch and updates the buffers, while evaluation uses the stored
    running statistics. ``norm="group_norm"`` normalizes ``groups`` channel
    groups per example and behaves identically in both modes; ``groups`` must
    divide the channel count. ``norm="none"`` gives a plain residual
    convolution block — no normalization at all — and in that case the
    convolutions carry a learned bias, which the normalized variants omit
    because the following normalization would cancel it.

    Convolutions and normalizations run in the plan's compute dtype; batch
    normalization accumulates its batch statistics in that dtype as PyTorch
    does, so ``float16`` plans inherit the usual reduced-precision behavior.
    """

    def __init__(self, out_channels, in_channels, stride, norm, groups):
        super().__init__()
        self.stride = stride
        self.norm = norm
        bias = norm == "none"
        self.conv1 = nn.Conv2d(in_channels, out_channels, KERNEL, stride=stride, padding=PADDING, bias=bias)
        self.norm1 = _norm(norm, out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, KERNEL, stride=1, padding=PADDING, bias=bias)
        self.norm2 = _norm(norm, out_channels, groups)
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias),
                                          _norm(norm, out_channels, groups))

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

    def extra_repr(self):
        return f"stride={self.stride}, norm={self.norm!r}"
