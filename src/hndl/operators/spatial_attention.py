import torch
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, operator


def _relation(s):
    reduction = s.args["reduction"]
    channels = None
    for port in ("x", "out"):
        shape = s.shape(port)
        if shape is not None and len(shape) == 4 and shape[1] is not None:
            channels = shape[1]
            break
    if channels is None:
        return
    if channels % reduction:
        s.error("E_CONSTRAINT", f"reduction={reduction} must divide the channel width C={channels}")


def _reference(module):
    """The same block contracted with ``torch.einsum`` instead of ``torch.bmm``."""
    def run(x):
        batch, channels, height, width = x.shape
        positions = height * width
        f = module.f_proj(x).reshape(batch, module.query_channels, positions)
        g = module.g_proj(x).reshape(batch, module.query_channels, positions)
        h = module.h_proj(x).reshape(batch, channels, positions)
        energy = torch.einsum("bcn,bcm->bnm", f, g)
        beta = energy.softmax(dim=-1)
        attended = torch.einsum("bcm,bnm->bcn", h, beta)
        return module.gamma * attended.reshape(batch, channels, height, width) + x
    return run


@operator(
    "spatial_attention",
    summary="SAGAN self-attention over every position of a [B, C, H, W] feature map.",
    shape="x[B, C, H, W] -> out[B, C, H, W]",
    relation=_relation,
    args={
        "reduction": Arg(int, 8, min=1,
                         help="How much narrower the query and key projections are than the input; must evenly "
                              "divide the channel width C. The SAGAN default of 8 gives C/8 query channels."),
    },
    reference=_reference,
    examples=[
        Example("spatial_attention()", ("B", 64, 8, 8), ("B", 64, 8, 8),
                "The SAGAN block at its paper settings: 64 channels, C/8 = 8 query channels, "
                "64 positions attending to each other."),
        Example("spatial_attention(2)", ("B", 16, 4, 4), ("B", 16, 4, 4),
                "A small feature map with a gentler reduction: 8 query channels out of 16."),
        Example("conv(32, kernel_size=3, padding=1)\nspatial_attention(4)\nglobal_avg_pool()\nlinear()",
                ("B", 3, 8, 8), ("B", 10),
                "A convolutional trunk with one attention block; the 32 channels flow into the block "
                "and out of it unchanged."),
    ],
    category="spatial",
)
class SpatialAttention(nn.Module):
    """The Self-Attention GAN block on a ``[B, C, H, W]`` feature map: every one
    of the ``N = H*W`` positions attends to every other, so the layer sees the
    whole map where a 3x3 convolution sees a neighbourhood.

    ```text
    f, g    = f_proj(x), g_proj(x)              # 1x1 convs, C -> C/reduction
    h       = h_proj(x)                         # 1x1 conv,  C -> C
    f, g, h -> [B, C', N], [B, C', N], [B, C, N]     # N = H*W
    energy[b, i, j] = sum_c f[b, c, i] * g[b, c, j]  # [B, N, N]
    beta    = softmax(energy, dim=-1)           # over the key positions j
    o[b, c, i] = sum_j h[b, c, j] * beta[b, i, j]    # [B, C, N] -> [B, C, H, W]
    out     = gamma * o + x
    ```

    ``reduction`` must divide ``C``; the resolver reports ``E_CONSTRAINT`` when
    it does not. Submodules are ``f_proj`` (query), ``g_proj`` (key) and
    ``h_proj`` (value), each an ``nn.Conv2d(C, ·, kernel_size=1)`` with a bias,
    so their parameters are ``f_proj.weight``, ``f_proj.bias`` and so on.
    ``f_proj`` and ``g_proj`` project to ``C // reduction`` channels while
    ``h_proj`` keeps all ``C``. Nothing else is learned: the softmax, the two
    contractions and the reshapes carry no state and there are no buffers.

    The remaining parameter is ``gamma``, a single learned scalar of shape
    ``[1]`` that gates the attention branch, exactly as `learned_scale` does.
    It starts at ``0.0``, so the block begins as the identity — ``out == x`` on
    the first step — and the network learns how much attention to admit. A
    gamma of zero also zeroes the gradient reaching the three projections;
    they start learning through ``gamma`` itself, whose gradient is the inner
    product of the attention branch with the incoming gradient. Construction
    overrides target it by name: ``spatial_attention(init={"gamma": 1.0})``
    opens the gate and ``spatial_attention(trainable={"gamma": False})``
    freezes it.

    Shape, rank and dtype are preserved and train and eval behave identically.
    Cost grows with ``N**2 = (H*W)**2``: the attention map is ``[B, N, N]``,
    which is 256x256 per example on a 16x16 map and 4096x4096 on a 64x64 one,
    so the block is normally placed at a middle resolution. ``H`` and ``W`` are
    not inferable backward — every spatial extent is accepted — so they come
    from the network input or the preceding operation, while ``C`` flows in
    both directions.

    Zhang, Goodfellow, Metaxas & Odena, "Self-Attention Generative Adversarial
    Networks" (ICML 2019, arXiv 2018), section 3: f and g project to
    ``C/8`` channels, h keeps ``C``, the attention map is
    ``softmax(f(x)^T g(x))`` over the key positions, and the output is
    ``y = gamma * o + x`` with gamma initialized to 0.
    """

    def __init__(self, reduction, *, C):
        super().__init__()
        if C % reduction:
            raise HNDLError("E_CONSTRAINT", f"reduction={reduction} must divide the channel width C={C}")
        self.reduction = reduction
        self.query_channels = C // reduction
        self.f_proj = nn.Conv2d(C, self.query_channels, kernel_size=1)
        self.g_proj = nn.Conv2d(C, self.query_channels, kernel_size=1)
        self.h_proj = nn.Conv2d(C, C, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.shape
        positions = height * width
        f = self.f_proj(x).reshape(batch, self.query_channels, positions)
        g = self.g_proj(x).reshape(batch, self.query_channels, positions)
        h = self.h_proj(x).reshape(batch, channels, positions)
        energy = torch.bmm(f.transpose(1, 2), g)
        beta = energy.softmax(dim=-1)
        attended = torch.bmm(h, beta.transpose(1, 2))
        return self.gamma * attended.reshape(batch, channels, height, width) + x

    def extra_repr(self):
        return f"reduction={self.reduction}, query_channels={self.query_channels}"
