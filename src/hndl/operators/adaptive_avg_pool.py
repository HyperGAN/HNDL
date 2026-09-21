import torch
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, PAIR, operator


def _relation(s):
    size = s.args.get("output_size")
    if size is not None:
        s.axis("out", 2, size[0])
        s.axis("out", 3, size[1])
        return
    shape = s.shape("out")
    if shape is not None and shape[2] is not None and shape[3] is not None:
        s.arg("output_size", (shape[2], shape[3]))


def _windows(extent, count):
    """The torch adaptive-pooling windows: ``[floor(i*extent/count), ceil((i+1)*extent/count))``."""
    return [(i * extent // count, -((-(i + 1) * extent) // count)) for i in range(count)]


def _reference(module):
    out_h, out_w = module.output_size

    def forward(x):
        rows = []
        for start_h, end_h in _windows(x.shape[2], out_h):
            columns = []
            for start_w, end_w in _windows(x.shape[3], out_w):
                columns.append(x[:, :, start_h:end_h, start_w:end_w].mean(dim=(2, 3)))
            rows.append(torch.stack(columns, dim=-1))
        return torch.stack(rows, dim=-2)

    return forward


@operator(
    "adaptive_avg_pool",
    identity="adaptive_avg_pool2d",
    summary="Average-pool [B, C, H, W] images to a fixed output height and width.",
    shape="x[B, C, H_in, W_in] -> out[B, C, H_out, W_out]",
    relation=_relation,
    reference=_reference,
    shape_text="H_out, W_out == output_size; C is preserved; H_in, W_in are unconstrained",
    args={
        "output_size": Arg(PAIR, inferable=True, min=1, max=MAX_DIMENSION_LITERAL,
                           help="Target output height and width; an int applies to both. "
                                "Omit it to read the target from the output contract."),
    },
    examples=[
        Example("conv(16, kernel_size=3, padding=1)\nrelu()\nadaptive_avg_pool(1)\nflatten()\nlinear()",
                ("B", 3, 28, 28), ("B", 10),
                "Pooling to 1x1 turns the feature map into one value per channel."),
        Example("adaptive_avg_pool(4)\nflatten()\nlinear()", ("B", 8, 15, 15), ("B", 5),
                "A ragged 15x15 map becomes a fixed 4x4 grid, so the classifier head has a fixed width."),
        Example("adaptive_avg_pool()", ("B", 6, 9, 9), ("B", 6, 3, 3),
                "The omitted output_size is inferred as (3, 3) from the output contract."),
        Example("adaptive_avg_pool((2, 3))", ("B", 4, 8, 8), ("B", 4, 2, 3),
                "A pair pools height and width to different extents."),
    ],
    category="spatial",
)
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Averages each ``[B, C, H_in, W_in]`` feature map over a grid of
    ``output_size = (H_out, W_out)`` windows, producing
    ``[B, C, H_out, W_out]``. Window ``i`` along an axis of extent ``n`` with
    ``k`` outputs covers

    ```text
    [floor(i * n / k), ceil((i + 1) * n / k))
    ```

    so the windows tile the axis, overlap when ``k`` does not divide ``n``,
    and repeat a single element when ``k > n`` (an upsampling by replication,
    which torch permits). The operation has no parameters and behaves
    identically in train and eval mode.

    Shape inference runs forward only for the spatial axes: ``H_out`` and
    ``W_out`` come from ``output_size``, but ``H_in`` and ``W_in`` are *not*
    inferable backward, because every input extent maps to the requested
    output. Give the input extents from the network input or the preceding
    operation. The channel count ``C`` is preserved and flows both ways, and
    ``output_size`` itself can be inferred backward from a known output shape.

    The average is accumulated in the input dtype; in ``float16`` a very large
    window loses precision, so pool in two stages if that matters.
    """

    def __init__(self, output_size):
        super().__init__(tuple(output_size))
