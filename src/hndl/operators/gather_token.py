import torch
from torch import nn

from ..operator import Arg, Example, operator

MODES = ("argmax",)


def _reference(module):
    """Advanced indexing instead of ``gather``: pick row ``argmax(ids[b])`` of ``x[b]``."""
    def run(x, ids):
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, torch.argmax(ids, dim=1)]

    return run


@operator(
    "gather_token",
    summary="Select the token of a sequence at the position holding the largest id.",
    shape="x[B, T, D], ids[B, T]:int64 -> out[B, D]",
    args={
        "mode": Arg(str, "argmax", choices=MODES, positional=False,
                    help="How the position is chosen from ids; only argmax is supported today."),
    },
    reference=_reference,
    examples=[
        Example("tokens = embedding(x, 32, 8)\nfeatures = linear(tokens, 8)\ngather_token(features, x)",
                ("B", 6), ("B", 8),
                "CLIP-style end-of-text pooling: the ids fan out to the embedding and to the gather.",
                input_dtype="int64"),
        Example("h = embedding(x, 64, 16)\nh = pos_embed(h, 32)\nh = transformer_block(h, 4)\n"
                "pooled = gather_token(h, x)\nlinear(pooled)",
                ("B", 10), ("B", 12),
                "A small text encoder: the pooled end-of-text feature is projected to the output width.",
                input_dtype="int64"),
    ],
    category="sequence",
)
class GatherToken(nn.Module):
    """Picks one position out of a ``[B, T, D]`` sequence per example and
    returns the ``[B, D]`` vector found there. Which position is chosen is
    decided by a second input, ``ids``, an int64 tensor of shape ``[B, T]``:

    ```text
    p[b]      = argmax_t ids[b, t]
    out[b, :] = x[b, p[b], :]
    ```

    This is CLIP's end-of-text pooling. A tokenizer writes the end-of-text
    marker as the highest id in the vocabulary, so the argmax of the id row
    lands on the final real token of that example and the padding after it is
    ignored. Unlike ``pool_tokens("last")``, which always takes position
    ``T - 1``, the selected position varies per example, which is what a
    padded batch needs.

    ``ids`` is data, not activations. It is typically the graph input itself,
    declared with ``input_dtype="int64"`` and fanned out to both
    ``embedding`` and this operator:

    ```python
    network('tokens = embedding(x, 32, 8)\\n'
            'features = linear(tokens, 8)\\n'
            'gather_token(features, x)',
            input_shape=("B", 6), output_shape=("B", 8),
            input_dtype="int64", device="cpu")
    ```

    Both inputs are explicit: neither port is filled by the implicit current
    tensor, so the configuration names the sequence and the ids. The two must
    agree on the batch and on the token count ``T``; a mismatch is a shape
    contradiction, reported as ``E_CONSTRAINT`` at resolution.

    Ties follow ``torch.argmax``: when a row holds its maximum id more than
    once, the first such position wins. An all-equal row therefore selects
    position 0. Ids are never bounds-checked against the vocabulary here,
    because only their ordering matters.

    ``mode`` exists so that other selection rules can be added later; today
    ``"argmax"`` is the only choice.

    There are no parameters and no state, so train and eval behave
    identically. The gather is exact in every compute dtype: values are moved,
    never combined. Gradients flow to ``x`` only, as a scatter that leaves
    every unselected position at zero; ``ids`` is integer and carries none.
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, ids):
        positions = torch.argmax(ids, dim=1)
        index = positions.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        return x.gather(1, index).squeeze(1)

    def extra_repr(self):
        return f"mode={self.mode!r}"
