import torch
from torch import nn

from ..operator import Example, operator

from ._relations import broadcast


def _relation(s):
    broadcast(s, "broadcast_add")

@operator(
    "broadcast_add",
    summary="Elementwise sum of two tensors of equal rank, broadcasting size-1 axes.",
    shape="a, b -> out",
    relation=_relation,
    shape_text="equal rank and batch; per non-batch axis the extents are equal or one of them is 1; "
               "out takes the larger extent",
    reference=lambda module: torch.add,
    examples=[
        Example("saved = x\nglobal_avg_pool()\nlinear(3)\nbias = reshape(3, 1, 1)\nbroadcast_add(saved, bias)",
                ("B", 3, 8, 8), ("B", 3, 8, 8),
                "A per-sample, per-channel conditioning bias: the pooled [B, 3, 1, 1] projection is "
                "added to every spatial position."),
        Example("h = conv(8, kernel_size=3, padding=1)\npooled = adaptive_avg_pool(h, 1)\nbroadcast_add(h, pooled)",
                ("B", 3, 8, 8), ("B", 8, 8, 8),
                "A squeeze-and-excite style shortcut: the 1x1 channel summary is added back to the map."),
        Example("h = linear(8)\nsummary = mean(h, 1, keepdim=True)\nbroadcast_add(h, summary)",
                ("B", 6, 4), ("B", 6, 8),
                "Sequences: a [B, 1, D] summary token is added to every position of a [B, T, D] tensor."),
    ],
    category="arithmetic",
)
class BroadcastAdd(nn.Module):
    """``out = a + b`` with numpy-style broadcasting restricted to operands of
    **equal rank**. Use this when one operand is a per-sample, per-channel
    bias such as ``[B, C, H, W] + [B, C, 1, 1]``; use `add` when the shapes
    match exactly, which is the stricter and cheaper contract.

    Broadcasting is symmetric: for every non-batch axis the two extents must
    either be equal or one of them must be 1, and the output takes the larger
    extent. Either operand may hold the size-1 axis, so ``[B, 1, H, W] +
    [B, C, 1, 1]`` is accepted and produces ``[B, C, H, W]``. The batch axis
    is never broadcast — both operands carry the full batch — and rank is
    never padded, so a ``[B, C, H, W]`` map and a ``[B, C]`` vector are a rank
    mismatch (``E_CONSTRAINT``); `reshape` the vector to ``[B, C, 1, 1]``
    first. Incompatible extents report ``E_CONSTRAINT`` naming the axis.

    Shape inference runs in both directions wherever the answer is unique. A
    known pair of inputs always fixes the output. Going backward, a known
    output plus one known operand fixes the other operand on every axis where
    the known operand has extent 1 (the other must supply the full extent) and
    on every axis where the output extent is 1 (both operands must be 1). On
    an axis where the known operand already matches a larger output extent the
    other operand could be either that extent or 1, so nothing is inferred
    there and the extent has to come from the operand's own producer.

    ### Determinism

    The backward pass reduces each broadcast axis with a plain ``sum``
    (torch's ``sum_to_size``), which is a deterministic tree reduction with no
    atomic accumulation — unlike ``index_add``/``scatter_add``, which are the
    usual source of run-to-run drift on CUDA. Both the forward and the
    backward run cleanly under
    ``torch.use_deterministic_algorithms(True)``, and repeated backward passes
    over identical inputs on CUDA give bit-identical gradients. The operator
    has no parameters and behaves identically in train and eval mode; the sum
    is computed in the plan compute dtype.
    """

    def forward(self, a, b):
        return a + b
