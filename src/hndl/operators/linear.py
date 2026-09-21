from torch import nn
from torch.nn.utils import parametrizations

from ..operator import Arg, Example, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    shape = s.shape("x") or s.shape("out")
    if shape is not None and len(shape) == 4:
        s.error("E_CONSTRAINT", "linear expects [B, D] or [B, T, D]; flatten or reshape image tensors first")


@operator(
    "linear",
    summary="Fully connected layer: a learned affine map on the last axis.",
    shape="x[B, ..., D_in] -> out[B, ..., D_out]",
    relation=_relation,
    args={
        "out_features": Arg(int, inferable=True, dim="D_out", min=1, max=MAX_DIMENSION_LITERAL,
                            help="Output width. Omit it to infer the width from what follows."),
        "in_features": Arg(int, inferable=True, dim="D_in", min=1, max=MAX_DIMENSION_LITERAL, positional=False,
                           help="Input width. Normally inferred from the incoming tensor."),
        "bias": Arg(bool, True, positional=False, help="Add a learned bias vector."),
        "spectral_norm": Arg(bool, False, positional=False,
                             help="Divide the weight by its largest singular value, estimated by power iteration."),
    },
    examples=[
        Example("linear(64)\nrelu()\nlinear()", ("B", 128), ("B", 10),
                "The final width is inferred from the output contract."),
        Example("linear(32, bias=False)", ("B", 16), ("B", 32)),
        Example("linear(64)\nrelu()\nlinear()", ("B", 16, 32), ("B", 16, 8),
                "On a [B, T, D] sequence the map applies to every position."),
        Example("linear(128, spectral_norm=True)\nleaky_relu(0.2)\nlinear(spectral_norm=True)",
                ("B", 256), ("B", 1),
                "A spectrally normalized GAN critic head: every layer is 1-Lipschitz by construction."),
    ],
    category="core",
)
class Linear(nn.Linear):
    """Computes ``out = x @ weight.T + bias`` with ``weight`` of shape
    ``[out_features, in_features]``, applied to the last axis of ``[B, D]``
    or ``[B, T, D]`` inputs. No activation is applied; add one explicitly.
    Parameters are ``weight`` and, when enabled, ``bias``.

    ## Spectral normalization

    With ``spectral_norm=True`` the weight is reparametrized as ``weight /
    sigma(weight)``, where ``sigma`` is the largest singular value estimated
    by one power iteration per forward pass
    (``torch.nn.utils.parametrizations.spectral_norm``). Each layer is then
    1-Lipschitz, which is the standard constraint for a GAN discriminator.

    The parametrization renames the registered state: the learned tensor
    becomes ``parametrizations.weight.original`` and ``weight`` turns into a
    computed attribute, with persistent buffers
    ``parametrizations.weight.0._u`` and ``parametrizations.weight.0._v``
    holding the power-iteration vectors. ``init`` and ``trainable``
    overrides must therefore target ``parametrizations.weight.original``
    instead of ``weight``; ``bias`` is unaffected. The power iteration
    refreshes the buffers in training mode only, so evaluation is a pure
    function of the stored state.
    """

    def __init__(self, in_features, out_features, bias, spectral_norm):
        super().__init__(in_features, out_features, bias)
        if spectral_norm:
            parametrizations.spectral_norm(self)
