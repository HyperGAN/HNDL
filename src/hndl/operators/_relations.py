"""Reusable shape relations for operators whose rules exceed the shape DSL.

Relations receive a :class:`hndl.operator.NodeView` and refine dimensions in
both directions. They run in every solver sweep, so they must only add facts.
"""


def conv_output(extent, kernel, stride, padding, dilation):
    return (extent + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def conv_transpose_output(extent, kernel, stride, padding, dilation, output_padding):
    return (extent - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def spatial(kind, x="x", out="out"):
    """Two-dimensional convolution relations for ``conv2d`` or ``conv_transpose2d``."""
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
            axis = j + 2
            k, stride, pad, dilation = (args[key][j] for key in ("kernel_size", "stride", "padding", "dilation"))
            i, o = a[axis], b[axis]
            if i is not None:
                if transpose:
                    result = conv_transpose_output(i, k, stride, pad, dilation, args["output_padding"][j])
                else:
                    result = conv_output(i, k, stride, pad, dilation)
                s.axis(out, axis, result)
            if o is not None:
                if transpose:
                    numerator = o + 2 * pad - dilation * (k - 1) - args["output_padding"][j] - 1
                    if numerator % stride:
                        s.error("E_CONSTRAINT", f"Target extent {o} requires a non-integer transpose-convolution input")
                    s.axis(x, axis, numerator // stride + 1)
                else:
                    lower = max(1, (o - 1) * stride - 2 * pad + dilation * (k - 1) + 1)
                    upper = o * stride - 2 * pad + dilation * (k - 1)
                    if lower > upper:
                        s.error("E_CONSTRAINT", "Convolution inverse has no positive input extent")
                    s.interval(x, axis, lower, upper)

    return relation


def elementwise_join(*ports, out="out"):
    """All listed input ports and the output share one exact shape."""

    def relation(s):
        for port in ports:
            s.equal(port, out)

    return relation


def broadcast(s, operation):
    """Numpy-style broadcasting restricted to equal rank and a shared batch axis.

    Every fact is added only when it follows uniquely from what is known, so
    the relation stays monotone across solver sweeps.
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
