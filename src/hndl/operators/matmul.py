import torch
from torch import nn

from ..operator import Example, operator


@operator(
    "matmul",
    summary="Batched matrix product of two rank-3 tensors.",
    shape="a[B, N, K], b[B, K, M] -> out[B, N, M]",
    reference=lambda module: torch.bmm,
    examples=[
        Example("k = transpose(1, 2)\nmatmul(x, k)", ("B", 6, 4), ("B", 6, 6),
                "The energy map of self-attention: a sequence times its own transpose."),
        Example("k = transpose(1, 2)\ne = matmul(x, k)\na = softmax(e, -1)\nmatmul(a, x)",
                ("B", 6, 4), ("B", 6, 4),
                "Dot-product attention without projections: scores, softmax, then a weighted sum of x."),
        Example("g = linear()\nt = transpose(x, 1, 2)\nmatmul(g, t)", ("B", 4, 6), ("B", 4, 4),
                "The shared inner dimension K flows backward: the projection width is inferred as 6."),
    ],
    category="arithmetic",
)
class MatMul(nn.Module):
    """``out[i] = a[i] @ b[i]`` for every batch element ``i``: the operation
    ``torch.bmm`` performs (``torch.matmul`` agrees at rank 3).

    ```text
    a[B, N, K] @ b[B, K, M] -> out[B, N, M]
    out[i, n, m] = sum_k a[i, n, k] * b[i, k, m]
    ```

    Both tensor inputs must be supplied explicitly, as with ``mul`` and
    ``add``; there is no implicit current-tensor operand for the second port.
    The inner dimension ``K`` is a shared symbol, so a known width on either
    side fixes the other and a mismatch is reported as ``E_CONSTRAINT`` while
    the plan resolves. The batch is shared: there is no broadcasting of the
    batch axis or of a missing axis.

    Only rank 3 is supported. That is the layout attention works in —
    ``[B, T, D]`` sequences, or an image feature map flattened to
    ``[B, C, H*W]`` — and it keeps the operation unambiguous. Reshape first if
    a tensor arrives at another rank: ``reshape(C)`` turns ``[B, C, H, W]``
    into ``[B, C, H*W]``, and ``reshape(C, H, W)`` turns it back. A rank-2 or
    rank-4 tensor on any port fails with ``E_CONSTRAINT`` rather than being
    silently folded or broadcast.

    There are no parameters and no buffers, train and eval behave identically,
    and the product runs in the plan compute dtype. Reduced-precision matrix
    multiplies accumulate in float32 on CUDA tensor cores, so float16 and
    bfloat16 results are close to, but not bitwise equal to, the same product
    done in full precision.
    """

    def forward(self, a, b):
        return torch.matmul(a, b)
