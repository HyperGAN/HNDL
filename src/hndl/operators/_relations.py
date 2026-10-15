"""Deprecated location of :mod:`hndl.relations`, kept so existing imports work.

Import shape relation helpers from ``hndl.relations``, the public module.
"""

from ..operator import SUPPORTED_RANKS
from ..relations import (broadcast, conv_axis, conv_input_range, conv_output,  # noqa: F401
                         conv_transpose_axis, conv_transpose_input, conv_transpose_output,
                         elementwise_join, spatial)

__all__ = [
    "broadcast",
    "conv_axis",
    "conv_input_range",
    "conv_output",
    "conv_transpose_axis",
    "conv_transpose_input",
    "conv_transpose_output",
    "elementwise_join",
    "non_image",
    "reduce_axis",
    "spatial",
]


def non_image(s):
    """Reject rank-4 image tensors for sequence or feature operators."""
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) == 4:
        s.error("E_CONSTRAINT", f"{s.spec.alias} expects [B, D] or [B, T, D]; flatten or reshape image tensors first")


def reduce_axis(s):
    """Shape relation for axis reduction operators like mean and sum."""
    op = s.spec.alias
    dim, keepdim = s.args["dim"], s.args["keepdim"]
    x_shape, out_shape = s.shape("x"), s.shape("out")
    rank = len(x_shape) if x_shape is not None else None
    if rank is None and out_shape is not None:
        rank = len(out_shape) if keepdim else len(out_shape) + 1
    if rank is None:
        return
    if dim >= rank:
        s.error("E_ARGUMENT", f"{op} dim {dim} is outside rank {rank}")
    out_rank = rank if keepdim else rank - 1
    if out_rank < min(SUPPORTED_RANKS):
        s.error("E_CONSTRAINT", f"{op} over dim {dim} of a rank-{rank} tensor would produce rank {out_rank}; "
                                "HNDL tensors keep a batch axis and at least one feature axis, so pass "
                                "keepdim=True or reduce a higher-rank tensor")
    if rank not in SUPPORTED_RANKS:
        s.error("E_CONSTRAINT", f"{op} input rank {rank} is unsupported; HNDL tensors have rank "
                                f"{', '.join(map(str, SUPPORTED_RANKS))}")
    s.rank("x", rank)
    s.rank("out", out_rank)
    x_shape, out_shape = s.shape("x"), s.shape("out")
    for axis in range(1, rank):
        if axis == dim:
            if keepdim:
                s.axis("out", dim, 1)
            continue
        target = axis if keepdim or axis < dim else axis - 1
        extent = x_shape[axis] if x_shape[axis] is not None else out_shape[target]
        s.axis("x", axis, extent)
        s.axis("out", target, extent)
