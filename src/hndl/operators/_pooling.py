"""Shared shape arithmetic for the two-dimensional pooling operators.

``max_pool`` and ``avg_pool`` agree on their geometry: the batch and channel
axes pass through untouched and each spatial axis shrinks by the windowing
rule ``floor((I + 2*padding - kernel_size) / stride) + 1`` with dilation 1 and
``ceil_mode`` false. A ``stride`` of 0 is the PyTorch default of "step by one
whole window", so it normalizes to ``kernel_size`` on that axis.
"""

from ..errors import HNDLError


def normalize_stride(stride, kernel_size):
    """Replace a 0 stride with the kernel extent on each axis."""
    return tuple(step or kernel for step, kernel in zip(stride, kernel_size))


def pool_output(extent, kernel, stride, padding):
    return (extent + 2 * padding - kernel) // stride + 1


def validate_padding(args):
    """Reject padding torch refuses: more than half the window on either axis."""
    for axis, (kernel, padding) in enumerate(zip(args["kernel_size"], args["padding"])):
        if padding > kernel // 2:
            name = "height" if axis == 0 else "width"
            raise HNDLError("E_ARGUMENT", f"padding {padding} on the {name} axis must be at most "
                                          f"half the window, kernel_size//2 = {kernel // 2}")


def pooling(x="x", out="out"):
    """Relate the spatial axes of a pooling node in both directions."""

    def relation(s):
        args = s.args
        a, b = s.shape(x), s.shape(out)
        for j in range(2):
            axis = j + 2
            kernel, padding = args["kernel_size"][j], args["padding"][j]
            stride = args["stride"][j] or kernel
            i = None if a is None else a[axis]
            o = None if b is None else b[axis]
            if i is not None:
                if i + 2 * padding < kernel:
                    s.error("E_CONSTRAINT", f"A {kernel}-wide window does not fit an extent of "
                                            f"{i} padded by {padding}")
                else:
                    s.axis(out, axis, pool_output(i, kernel, stride, padding))
            if o is not None:
                lower = max(1, (o - 1) * stride - 2 * padding + kernel)
                upper = o * stride - 2 * padding + kernel - 1
                if lower > upper:
                    s.error("E_CONSTRAINT", "Pooling inverse has no positive input extent")
                else:
                    s.interval(x, axis, lower, upper)

    return relation
