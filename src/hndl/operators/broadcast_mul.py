import torch
from torch import nn

from ..operator import Example, operator
from ..relations import broadcast


def _relation(s):
    broadcast(s, "broadcast_mul")


@operator(
    "broadcast_mul",
    summary="Elementwise product of equal-rank tensors, broadcasting size-1 axes.",
    shape="a, b -> out",
    relation=_relation,
    shape_text="equal rank and batch; each non-batch axis is equal or size 1; output takes the larger extent",
    reference=lambda module: torch.mul,
    examples=[
        Example('h = linear(8)\ng = mean(h, 1, keepdim=True)\nbroadcast_mul(h, g)',
                ("B", 6, 4), ("B", 6, 8), "A per-image channel gate scales every token."),
    ],
    category="arithmetic",
)
class BroadcastMul(nn.Module):
    """Compute ``a * b`` with equal-rank, non-batch broadcasting.

    For example, ``[B,T,D] * [B,1,D]`` applies one channel gate per image.
    Both operands must have the same batch and rank; reshape style vectors
    explicitly. Gradients sum over broadcast axes, including for higher
    derivatives. No parameters or buffers; train and eval are identical.
    Use ``mul`` when broadcasting should be forbidden.
    """

    def forward(self, a, b):
        return a * b
