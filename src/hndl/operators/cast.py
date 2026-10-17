"""Explicit dtype conversion across graph edges."""

import torch
from torch import nn

from ..operator import Arg, Example, operator

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def _reference(module):
    target = module.target_dtype

    def run(x):
        dt = module._sink.dtype if target == "compute" else _TORCH_DTYPES[target]
        return x.to(dt)

    return run


@operator(
    "cast",
    summary="Explicit dtype conversion across graph edges.",
    shape="x[B, ...]:any -> out[B, ...]:any",
    args={
        "dtype": Arg(
            str,
            "compute",
            choices=("compute", "float32", "float16", "bfloat16", "float64", "int64", "int32", "bool"),
            help="Target tensor dtype; 'compute' targets the plan compute dtype.",
        ),
    },
    examples=[
        Example("cast('compute')", ("B", 16), ("B", 16), "Cast features to the plan compute dtype."),
    ],
    reference=_reference,
    category="core",
)
class Cast(nn.Module):
    """Cast a tensor to a target data type across graph edges.

    Converts an input tensor along the edge to the specified target data type.
    When ``dtype`` is set to ``"compute"``, the tensor is cast to the active
    plan compute dtype (e.g. ``float32``, ``float16``, or ``bfloat16``).
    """

    def __init__(self, dtype="compute"):
        super().__init__()
        self.target_dtype = dtype
        self.register_buffer("_sink", torch.empty(0))

    def forward(self, x):
        target = self._sink.dtype if self.target_dtype == "compute" else _TORCH_DTYPES[self.target_dtype]
        return x.to(target)
