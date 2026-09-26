"""Reusable shape relations for operators whose rules exceed the shape DSL.

This module is public and stable: built-in operators use exactly these
helpers, and an operator declared outside HNDL may use them in its own
``relation=`` function instead of reimplementing convolution arithmetic.

There are two kinds of helper:

- **Arithmetic** on plain integers: :func:`conv_output`,
  :func:`conv_input_range`, :func:`conv_transpose_output` and
  :func:`conv_transpose_input`. They follow PyTorch's ``Conv*d`` and
  ``ConvTranspose*d`` output-size formulas and know nothing about nodes.
- **Relations** that receive the node view ``s`` a ``relation=`` function
  gets (:class:`hndl.operator.NodeView`) and add facts to it:
  :func:`conv_axis`, :func:`conv_transpose_axis`, :func:`spatial`,
  :func:`elementwise_join` and :func:`broadcast`.

A relation runs in every solver sweep, in whatever order the solver visits
nodes, so it must be monotone: it may only add facts that follow from what
is already known, never retract or guess one. Every relation here works in
both directions, so a known output extent constrains the input as well as
the other way round.

Example: a ``[B, C, L]`` operator that keeps every ``window``-th position
reuses the convolution rule on its length axis::

    from hndl.relations import conv_axis

    def relation(s):
        s.rank("x", 3)
        s.rank("out", 3)
        channels = s.shape("x")[1] or s.shape("out")[1]
        s.axis("x", 1, channels)
        s.axis("out", 1, channels)
        conv_axis(s, 2, kernel=1, stride=s.args["window"])

``docs/ADDING_OPERATORS.md`` declares that operator in full.
"""

from .errors import HNDLError

__all__ = ["broadcast", "conv_axis", "conv_input_range", "conv_output", "conv_transpose_axis",
           "conv_transpose_input", "conv_transpose_output", "elementwise_join", "spatial"]


# -- arithmetic ----------------------------------------------------------------------------


def conv_output(extent, kernel, stride=1, padding=0, dilation=1):
    """The output extent of a convolution or pooling window along one axis.

    ``floor((extent + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1``,
    the formula of ``torch.nn.Conv1d/2d/3d`` and of max and average pooling
    without ``ceil_mode``. The result can be zero or negative when the window
    does not fit; callers that need a positive extent check it themselves.
    """
    return (extent + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def conv_input_range(extent, kernel, stride=1, padding=0, dilation=1):
    """Every input extent a convolution maps to ``extent``, as ``(lower, upper)``.

    :func:`conv_output` floors a division, so with ``stride > 1`` several
    input extents give the same output; they form the inclusive interval
    returned here. ``lower`` is at least 1. ``lower > upper`` means no
    positive input extent produces ``extent``. Pass the pair to
    ``s.interval(port, axis, lower, upper)`` so the resolver narrows the input
    without choosing one value for it.
    """
    lower = max(1, (extent - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1)
    upper = extent * stride - 2 * padding + dilation * (kernel - 1)
    return lower, upper


def conv_transpose_output(extent, kernel, stride=1, padding=0, dilation=1, output_padding=0):
    """The output extent of a transposed convolution along one axis.

    ``(extent - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1``,
    the formula of ``torch.nn.ConvTranspose1d/2d/3d``.
    """
    return (extent - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def conv_transpose_input(extent, kernel, stride=1, padding=0, dilation=1, output_padding=0):
    """The one input extent a transposed convolution maps to ``extent``, or None.

    The transposed formula is injective, so the inverse is exact. None means
    ``extent`` is not reachable from any integer input extent with these
    arguments.
    """
    numerator = extent + 2 * padding - dilation * (kernel - 1) - output_padding - 1
    if numerator % stride:
        return None
    return numerator // stride + 1


# -- relations -----------------------------------------------------------------------------


def conv_axis(s, axis, kernel, stride=1, padding=0, dilation=1, *, x="x", out="out"):
    """Relate one axis of ``x`` and ``out`` by :func:`conv_output`, in both directions.

    A known input extent fixes the output extent. A known output extent
    narrows the input to :func:`conv_input_range`; with ``stride > 1`` that
    is an interval, which the resolver reports as ambiguous rather than
    picking a value unless another relation fixes it. An output extent no
    positive input reaches fails with ``E_CONSTRAINT``.

    Call it once per strided axis from a ``relation=`` function. It does not
    set ranks or touch other axes: pair it with ``s.rank`` and ``s.equal`` or
    ``s.axis`` for the rest of the shape.
    """
    a, b = s.shape(x), s.shape(out)
    if a is not None and a[axis] is not None:
        s.axis(out, axis, conv_output(a[axis], kernel, stride, padding, dilation))
    if b is not None and b[axis] is not None:
        lower, upper = conv_input_range(b[axis], kernel, stride, padding, dilation)
        if lower > upper:
            s.error("E_CONSTRAINT", "Convolution inverse has no positive input extent")
        s.interval(x, axis, lower, upper)


def conv_transpose_axis(s, axis, kernel, stride=1, padding=0, dilation=1, output_padding=0, *, x="x", out="out"):
    """Relate one axis of ``x`` and ``out`` by :func:`conv_transpose_output`, in both directions.

    Both directions are exact. An output extent that no integer input extent
    reaches fails with ``E_CONSTRAINT``.
    """
    a, b = s.shape(x), s.shape(out)
    if a is not None and a[axis] is not None:
        s.axis(out, axis, conv_transpose_output(a[axis], kernel, stride, padding, dilation, output_padding))
    if b is not None and b[axis] is not None:
        extent = conv_transpose_input(b[axis], kernel, stride, padding, dilation, output_padding)
        if extent is None:
            s.error("E_CONSTRAINT", f"Target extent {b[axis]} requires a non-integer transpose-convolution input")
        s.axis(x, axis, extent)


def spatial(kind, x="x", out="out"):
    """The complete relation of a 2D convolution, ``kind`` ``"conv2d"`` or ``"conv_transpose2d"``.

    Returns a function to pass as ``relation=``. Both ports are rank 4,
    ``[B, C, H, W]``. The operator must declare the arguments ``in_channels``
    and ``out_channels`` (``int``, ``inferable=True``), ``groups`` (``int``),
    and ``kernel_size``, ``stride``, ``padding`` and ``dilation`` as pairs
    (``"pair"``); ``"conv_transpose2d"`` also needs a pair ``output_padding``.
    Channels are read from and written back to those arguments, ``groups``
    must divide both channel counts, and each spatial axis follows
    :func:`conv_axis` or :func:`conv_transpose_axis`.
    """
    if kind not in ("conv2d", "conv_transpose2d"):
        raise HNDLError("E_REGISTRY", f"spatial() kind must be 'conv2d' or 'conv_transpose2d', not {kind!r}")
    transpose = kind == "conv_transpose2d"

    def relation(s):
        args = s.args
        s.rank(x, 4)
        s.rank(out, 4)
        s.axis(x, 1, args.get("in_channels"))
        s.axis(out, 1, args.get("out_channels"))
        a, b = s.shape(x), s.shape(out)
        s.arg("in_channels", a[1])
        s.arg("out_channels", b[1])
        for channels in (a[1], b[1]):
            if channels is not None and channels % args["groups"]:
                s.error("E_CONSTRAINT", f"groups={args['groups']} must divide input and output channels")
        for j in range(2):
            window = [args[key][j] for key in ("kernel_size", "stride", "padding", "dilation")]
            if transpose:
                conv_transpose_axis(s, j + 2, *window, args["output_padding"][j], x=x, out=out)
            else:
                conv_axis(s, j + 2, *window, x=x, out=out)

    return relation


def elementwise_join(*ports, out="out"):
    """A relation in which every listed input port and ``out`` share one exact shape.

    ``relation=elementwise_join("a", "b")`` is the rule of an elementwise
    binary operation without broadcasting.
    """

    def relation(s):
        for port in ports:
            s.equal(port, out)

    return relation


def broadcast(s, operation):
    """Numpy-style broadcasting of ports ``a`` and ``b`` into ``out``.

    Restricted to equal rank and a shared batch axis: rank is never padded
    and axis 0 is never broadcast. ``operation`` names the operator in error
    messages. Call it from a relation, ``relation=lambda s: broadcast(s,
    "my_op")``.

    Every fact is added only when it follows uniquely from what is known, so
    the relation stays monotone across solver sweeps: an output extent with
    one operand equal to it leaves the other operand's extent (equal or 1)
    unknown.
    """
    ports = ("a", "b", "out")
    rank = next((len(shape) for shape in (s.shape(port) for port in ports) if shape is not None), None)
    if rank is None:
        return
    # A rank disagreement surfaces here as an E_CONSTRAINT rank conflict.
    for port in ports:
        s.rank(port, rank)
    shapes = {port: s.shape(port) for port in ports}
    for axis in range(1, rank):
        a, b, out = (shapes[port][axis] for port in ports)
        if a is not None and b is not None:
            if a != b and a != 1 and b != 1:
                s.error("E_CONSTRAINT",
                        f"{operation} axis {axis}: extents {a} (a) and {b} (b) do not broadcast; "
                        "the extents must be equal or one of them must be 1")
            s.axis("out", axis, max(a, b))
            continue
        if out is None:
            continue
        if out == 1:
            # Both operands must be 1 for the result to be 1.
            s.axis("a", axis, 1)
            s.axis("b", axis, 1)
            continue
        for known, other in (("a", "b"), ("b", "a")):
            extent = shapes[known][axis]
            if extent is None:
                continue
            if extent != 1 and extent != out:
                s.error("E_CONSTRAINT",
                        f"{operation} axis {axis}: extent {extent} ({known}) cannot broadcast to "
                        f"the output extent {out}; it must equal {out} or be 1")
            if extent == 1:
                # The other operand alone has to supply the output extent.
                s.axis(other, axis, out)
            # extent == out leaves the other operand ambiguous (out or 1); say nothing.
