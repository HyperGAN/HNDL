"""Numerics, shape arithmetic, and error codes for ``max_pool``."""

import pytest
import torch
import torch.nn.functional as F

from hndl import HNDLError, resolve
from hndl.torch import build


def extent(size, kernel, stride, padding):
    return (size + 2 * padding - kernel) // stride + 1


def pooled(source, input_shape, output_shape, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    return plan, build(plan, device=device, initialization_seed=1)


@pytest.mark.parametrize("kernel,stride,padding,size", [
    (2, 0, 0, 16),
    (3, 2, 1, 15),
    (2, 1, 0, 9),
    (4, 4, 2, 12),
    (5, 3, 2, 17),
])
def test_matches_torch_forward_and_backward(kernel, stride, padding, size, device):
    step = stride or kernel
    out = extent(size, kernel, step, padding)
    source = f"max_pool({kernel}, stride={stride}, padding={padding})"
    _, model = pooled(source, ("B", 3, size, size), ("B", 3, out, out), device)
    x = torch.randn(4, 3, size, size, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.max_pool2d(mirror, kernel, stride=step, padding=padding)
    assert tuple(actual.shape) == (4, 3, out, out)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_rectangular_window_and_stride(device):
    source = "max_pool((2, 3), stride=(2, 1), padding=(1, 1))"
    _, model = pooled(source, ("B", 2, 8, 9), ("B", 2, 5, 9), device)
    x = torch.randn(3, 2, 8, 9, device=device)
    expected = F.max_pool2d(x, (2, 3), stride=(2, 1), padding=(1, 1))
    torch.testing.assert_close(model(x=x)["output"], expected)


def test_zero_stride_means_kernel_size():
    default = resolve("max_pool(3)", input_shape=("B", 4, 12, 12), output_shape=("B", 4, 4, 4))
    explicit = resolve("max_pool(3, stride=3)", input_shape=("B", 4, 12, 12), output_shape=("B", 4, 4, 4))
    node, same = default.nodes[0], explicit.nodes[0]
    assert node.args["stride"] == (0, 0) and same.args["stride"] == (3, 3)
    assert node.output_shapes["out"] == same.output_shapes["out"] == ("B", 4, 4, 4)
    model = build(default, device="cpu", initialization_seed=1)
    assert model[node.id].stride == (3, 3)


def test_pooling_has_no_parameters():
    plan = resolve("max_pool(2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 4, 4))
    model = build(plan, device="cpu", initialization_seed=1)
    assert list(model[plan.nodes[0].id].parameters()) == []


def test_forward_inference_preserves_channels():
    plan = resolve("conv(6, kernel_size=3, padding=1)\nmax_pool(2)", input_shape=("B", 3, 16, 16),
                   output_shape=("B", 6, 8, 8))
    assert plan.nodes[-1].input_shapes["x"] == ("B", 6, 16, 16)
    assert plan.nodes[-1].output_shapes["out"] == ("B", 6, 8, 8)


def test_backward_inference_resolves_a_unique_input_extent():
    plan = resolve("reshape(4)\nmax_pool(2, stride=1)", input_shape=("B", 256), output_shape=("B", 4, 7, 7))
    assert plan.nodes[-1].input_shapes["x"] == ("B", 4, 8, 8)


def test_backward_inference_reports_an_ambiguous_interval():
    with pytest.raises(HNDLError) as error:
        resolve("reshape(4)\nmax_pool(2)", input_shape=("B", 256), output_shape=("B", 4, 4, 4))
    assert error.value.code == "E_AMBIGUOUS"
    assert "(8, 9)" in error.value.message


@pytest.mark.parametrize("source", ["max_pool(2, padding=2)", "max_pool(3, padding=2)",
                                    "max_pool((2, 4), padding=(1, 3))"])
def test_padding_larger_than_half_the_window_is_rejected(source):
    with pytest.raises(HNDLError) as error:
        resolve(source, input_shape=("B", 3, 16, 16), output_shape=("B", 3, 8, 8))
    assert error.value.code == "E_ARGUMENT"


def test_window_larger_than_the_padded_input_is_rejected():
    with pytest.raises(HNDLError) as error:
        resolve("max_pool(5)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 1, 1))
    assert error.value.code == "E_CONSTRAINT"


def test_contradicting_output_extent_is_rejected():
    with pytest.raises(HNDLError) as error:
        resolve("max_pool(2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 5, 5))
    assert error.value.code == "E_CONSTRAINT"


@pytest.mark.parametrize("value", [0, -1])
def test_kernel_size_must_be_positive(value):
    with pytest.raises(HNDLError) as error:
        resolve(f"max_pool({value})", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 4, 4))
    assert error.value.code == "E_ARGUMENT"
