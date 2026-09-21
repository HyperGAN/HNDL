from torch import nn

from ..operator import Arg, Example, operator


@operator(
    "scale",
    summary="Multiply a tensor by a fixed scalar.",
    shape="x[B, ...] -> out[B, ...]",
    args={"factor": Arg(float, help="Constant every element is multiplied by; it is not learned.")},
    reference=lambda module: (lambda x: x * module.factor),
    examples=[
        Example('saved = x\nlinear(8, name="branch")\nrelu()\nh = linear(4)\ns = scale(h, 0.5)\nadd(s, saved)',
                ("B", 4), ("B", 4), "A residual whose branch is damped by 0.5 before the sum."),
        Example("linear(8)\nscale(2.0)\ntanh()", ("B", 5, 4), ("B", 5, 8),
                "Sequences: the factor applies to every position and feature."),
        Example("conv(4, kernel_size=3, padding=1)\nscale(0.1)\nrelu()", ("B", 3, 8, 8), ("B", 4, 8, 8),
                "Images: a constant gain on the convolution output."),
    ],
    category="arithmetic",
)
class Scale(nn.Module):
    """``out = factor * x``: a shape-, rank-, and dtype-preserving constant
    gain with no parameters and no buffers. ``factor`` is a plan constant, not
    a learned value, so the gradient is ``factor * grad_out`` and the behavior
    is identical in train and eval mode. The multiplication happens in the
    input dtype; a large ``factor`` can overflow float16, so keep the product
    inside the representable range for the plan's compute dtype.
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor

    def extra_repr(self):
        return f"factor={self.factor}"
