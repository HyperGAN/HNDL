from torch import nn
from torch.nn import functional as F

from ..errors import HNDLError
from ..operator import Arg, Example, INTS, MAX_DIMENSION_LITERAL, operator

MODES = ("constant", "reflect", "replicate")


def _amounts(padding, rank):
    """Total padding added to each axis, indexed by axis."""
    totals = [0] * rank
    for index in range(len(padding) // 2):
        totals[rank - 1 - index] = padding[2 * index] + padding[2 * index + 1]
    return totals


def _relation(s):
    padding, mode = s.args["padding"], s.args["mode"]
    known = s.shape("x") or s.shape("out")
    if known is None:
        return
    rank = len(known)
    s.rank("x", rank)
    s.rank("out", rank)
    pairs = len(padding) // 2
    if pairs > rank - 1:
        s.error("E_ARGUMENT", f"padding describes {pairs} axes but a rank-{rank} tensor has only {rank - 1} "
                              "non-batch axes; the batch axis is never padded")
    if mode != "constant" and (rank not in (3, 4) or pairs != rank - 2):
        s.error("E_ARGUMENT", f"mode={mode!r} pads the spatial axes only: give 2 values for a rank-3 [B, C, L] "
                              f"tensor or 4 for a rank-4 [B, C, H, W] tensor, got {len(padding)} for rank {rank}")
    totals = _amounts(padding, rank)
    x, out = s.shape("x"), s.shape("out")
    for axis in range(1, rank):
        if x[axis] is not None:
            s.axis("out", axis, x[axis] + totals[axis])
        if out[axis] is not None:
            extent = out[axis] - totals[axis]
            if extent < 1:
                s.error("E_CONSTRAINT", f"Output extent {out[axis]} at axis {axis} is not larger than the "
                                        f"{totals[axis]} padded positions")
            s.axis("x", axis, extent)
    if mode == "reflect":
        x = s.shape("x")
        for index in range(pairs):
            axis = rank - 1 - index
            widest = max(padding[2 * index], padding[2 * index + 1])
            if x[axis] is not None and widest >= x[axis]:
                s.error("E_CONSTRAINT", f"reflect padding {widest} must be smaller than the input extent "
                                        f"{x[axis]} at axis {axis}")


def _validate(args):
    padding = args["padding"]
    if not padding or len(padding) % 2:
        raise HNDLError("E_ARGUMENT", f"padding must be a non-empty even number of values, one (left, right) pair "
                                      f"per padded axis, got {len(padding)}")
    if args["mode"] != "constant" and args["value"] != 0.0:
        raise HNDLError("E_ARGUMENT", f"value applies to mode='constant' only, not mode={args['mode']!r}")


def _reference(module):
    def pad(x):
        if module.mode == "constant":
            return F.pad(x, module.padding, mode="constant", value=module.value)
        return F.pad(x, module.padding, mode=module.mode)
    return pad


@operator(
    "pad",
    summary="Enlarge trailing axes with constant, reflected, or replicated borders.",
    shape="x -> out",
    relation=_relation,
    shape_text="out[axis] == x[axis] + left + right for each padded axis; all other axes equal",
    args={
        "padding": Arg(INTS, (), min=0, max=MAX_DIMENSION_LITERAL,
                       help="(left, right) pairs starting at the last axis, as torch.nn.functional.pad "
                            "orders them. Non-negative; two values per padded axis."),
        "value": Arg(float, 0.0, positional=False, help="Fill value for mode='constant'."),
        "mode": Arg(str, "constant", positional=False, choices=MODES,
                    help="Border rule: 'constant' fills with value, 'reflect' mirrors without repeating the "
                         "edge, 'replicate' repeats the edge."),
    },
    positional_rest="padding",
    validate=_validate,
    reference=_reference,
    examples=[
        Example("pad(1, 1, 1, 1)\nconv(4, kernel_size=3)", ("B", 3, 8, 8), ("B", 4, 8, 8),
                "One column on each side of W and one row on each side of H, so the 3x3 convolution "
                "preserves the image size."),
        Example('pad(2, 2, mode="reflect")', ("B", 3, 8), ("B", 3, 12),
                "Mirrors two positions onto each end of L in a [B, C, L] signal."),
        Example("linear()\npad(1, 1)", ("B", 16), ("B", 10),
                "Backward inference: the projection width 8 follows from the padded output."),
        Example('pad(0, 1, 0, 1, value=1.0, mode="constant")', ("B", 2, 4, 4), ("B", 2, 5, 5),
                "A one-position border of ones on the right and bottom edges."),
    ],
    category="shape",
)
class Pad(nn.Module):
    """``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

    ``padding`` lists ``(left, right)`` pairs **starting from the last axis**,
    exactly as ``F.pad`` orders them:

    - rank 2 ``[B, F]``: ``pad(left, right)`` widens the feature axis ``F``.
    - rank 3 ``[B, C, L]`` (equivalently ``[B, T, D]``): ``pad(left, right)``
      widens the last axis ``L``.
    - rank 4 ``[B, C, H, W]``: ``pad(left, right, top, bottom)`` widens ``W``
      first and then ``H``.

    The batch axis is never padded, so ``padding`` holds at most
    ``rank - 1`` pairs; anything longer, or an odd number of values, is
    ``E_ARGUMENT``. Each padded axis grows by ``left + right`` and every other
    axis is unchanged, and the relation runs in both directions: a known input
    extent fixes the output, and a known output extent fixes the input.

    ``mode='constant'`` fills the new positions with ``value`` and works for
    every supported rank. ``mode='reflect'`` and ``mode='replicate'`` follow
    the PyTorch restriction to spatial padding: a rank-3 tensor padded on its
    last axis, or a rank-4 tensor padded on ``H`` and ``W``; other
    rank/padding combinations are rejected with ``E_ARGUMENT``. ``reflect``
    additionally requires each pad width to be smaller than the extent it
    mirrors (``E_CONSTRAINT``), because the border itself is not repeated.
    ``value`` must stay at its default for the non-constant modes, which
    PyTorch does not accept.

    There are no parameters, behavior is identical in train and eval mode, the
    output keeps the input dtype, and the gradient of the padded positions is
    discarded (``constant``) or accumulated back onto the mirrored or repeated
    source positions (``reflect``, ``replicate``).
    """

    def __init__(self, padding, value, mode):
        super().__init__()
        self.padding = tuple(padding)
        self.value = value
        self.mode = mode

    def forward(self, x):
        if self.mode == "constant":
            return F.pad(x, self.padding, mode="constant", value=self.value)
        return F.pad(x, self.padding, mode=self.mode)

    def extra_repr(self):
        return f"padding={self.padding}, value={self.value}, mode={self.mode!r}"
