import torch
from torch import nn

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _reference(module):
    weight = module.weight

    def embed(ids):
        return torch.nn.functional.embedding(ids, weight)

    return embed


@operator(
    "embedding",
    summary="Look up a learned vector for every integer token id.",
    shape="ids[B, T]:int64 -> out[B, T, D]",
    args={
        "num_embeddings": Arg(int, min=1, max=MAX_DIMENSION_LITERAL,
                              help="Vocabulary size; ids must lie in [0, num_embeddings)."),
        "dim": Arg(int, inferable=True, dim="D", min=1, max=MAX_DIMENSION_LITERAL,
                   help="Width of each embedding vector. Omit it to infer the width from what follows."),
    },
    examples=[
        Example("embedding(1000, 32)\nlinear(8)", ("B", 16), ("B", 16, 8),
                "A vocabulary of 1000 tokens embedded into 32 features, then projected to 8.",
                input_dtype="int64"),
        Example("embedding(64)\nrelu()", ("B", 8), ("B", 8, 5),
                "The embedding width 5 is inferred from the output contract.",
                input_dtype="int64"),
        Example("embedding(128, 16)\ncls_token()\npos_embed(32)\nlinear(4)", ("B", 6), ("B", 7, 4),
                "A token embedding followed by a class token and learned positions.",
                input_dtype="int64"),
    ],
    category="sequence",
    reference=_reference,
)
class Embedding(nn.Embedding):
    """A learned lookup table: row ``i`` of ``weight`` is the vector for token
    id ``i``.

    ```text
    out[b, t, :] = weight[ids[b, t], :]
    ```

    The input port ``ids`` carries **int64** values, not activations, so the
    graph input must be declared with ``input_dtype="int64"`` when
    ``embedding`` is the first operation:

    ```python
    network("embedding(1000, 32)\\nlinear(8)", input_shape=("B", 16),
            output_shape=("B", 16, 8), input_dtype="int64", device="cpu")
    ```

    Ids are positions on the ``T`` axis of a ``[B, T]`` tensor and the result
    is the ``[B, T, D]`` sequence those tokens embed to. Every id must satisfy
    ``0 <= id < num_embeddings``; an out-of-range id is a CUDA-side or
    CPU-side indexing fault, not a shape error, because ids are data.

    The single parameter is ``weight`` of shape ``[num_embeddings, dim]``,
    initialized from the standard normal distribution. It is dense: the whole
    table receives a gradient contribution on every step. There is no padding
    index and no norm clipping; train and eval behave identically.
    """

    def __init__(self, num_embeddings, dim):
        super().__init__(num_embeddings, dim)
