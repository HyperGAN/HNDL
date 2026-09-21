import torch
from torch import nn

from ..errors import HNDLError
from ..operator import Arg, Example, INTS, MAX_DIMENSION_LITERAL, operator


def _relation(s):
    shape = s.args["shape"]
    if not shape:
        # An omitted shape is read from the output contract by finalize.
        return
    s.rank("out", len(shape) + 1)
    for axis, value in enumerate(shape, 1):
        s.axis("out", axis, value)


def _validate(args):
    if len(args["shape"]) > 3:
        raise HNDLError("E_ARGUMENT", "constant shape must have at most three non-batch dimensions")


def _finalize(args, input_shapes, output_shapes):
    return {**args, "shape": tuple(output_shapes["out"][1:])}


def _reference(module):
    shape, value, dtype = module.shape, module.value, module.dtype

    def constant(x):
        return torch.full((x.shape[0], *shape), value, dtype=dtype, device=x.device)

    return constant


@operator(
    "constant",
    summary="Emit a tensor of a fixed shape filled with one constant value.",
    shape="x[B, ...]:any -> out",
    relation=_relation,
    shape_text="out == [B, *shape]; x contributes only its batch extent and device",
    args={
        "shape": Arg(INTS, (), min=1, max=MAX_DIMENSION_LITERAL,
                     help="Non-batch dimensions of the constant. Omit it to read them "
                          "from the output contract."),
        "value": Arg(float, 0.0, positional=False, help="Value every element is filled with."),
    },
    positional_rest="shape",
    validate=_validate,
    finalize=_finalize,
    reference=_reference,
    examples=[
        Example("z = constant(16)\nconcat(x, z)\nlinear(8)", ("B", 4), ("B", 8),
                "A null conditioning vector of 16 zeros appended to the features."),
        Example("c = constant(3, 8, 8, value=0.5)\nsub(x, c)\nconv(4, kernel_size=3, padding=1)",
                ("B", 3, 8, 8), ("B", 4, 8, 8),
                "A constant image plane: every pixel of every channel is centered by 0.5."),
        Example("h = linear(8)\nz = constant(value=1.0)\nadd(h, z)", ("B", 4), ("B", 8),
                "The omitted shape is read from the [B, 8] contract the add imposes."),
    ],
    category="arithmetic",
)
class Constant(nn.Module):
    """``out = full([B, *shape], value)``: a source of fixed values with no
    parameters and no buffers.

    ```text
    out[b, ...] = value
    ```

    Give the non-batch dimensions positionally, ``constant(3, 8, 8)``, or as
    ``shape=(3, 8, 8)``; omit them entirely and the plan reads them from the
    output contract, the way ``reshape()`` does. ``value`` defaults to ``0.0``,
    so ``constant(16)`` is a zero vector — a null conditioning input — and
    ``constant(16, value=1.0)`` is a vector of ones.

    The incoming tensor ``x`` supplies only the runtime batch extent and the
    device; its own shape, rank, and values are ignored, so the input port
    accepts any dtype including an integer graph input such as token ids. The
    result always carries the plan's compute dtype and it is created fresh on
    every call, detached from the autograd graph: it requires no gradient and
    no gradient reaches ``x``. The operation is deterministic and identical in
    train and eval mode.

    Because the constant tells the solver nothing about ``x``, it does not
    propagate shapes backward: the operation before it must have its shape
    fixed from the input side or by its own arguments.
    """

    def __init__(self, shape, value):
        super().__init__()
        self.shape = tuple(shape)
        self.value = value
        # Unregistered so the node stays stateless; _apply keeps it in step
        # with the dtype and device the plan casts this module to.
        self._prototype = torch.empty(0)

    @property
    def dtype(self):
        """The compute dtype the plan built this node with."""
        return self._prototype.dtype

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._prototype = fn(self._prototype)
        return self

    def forward(self, x):
        return torch.full((x.shape[0], *self.shape), self.value,
                          dtype=self._prototype.dtype, device=x.device)

    def extra_repr(self):
        return f"shape={self.shape}, value={self.value}"
