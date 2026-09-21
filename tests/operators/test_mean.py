"""Numerics, bidirectional inference, and error codes for the ``mean`` reduction."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

from .conftest import DEVICES

REFERENCE = torch.mean


def reduced(shape, dim, keepdim):
    """The declared output contract of reducing ``shape`` along ``dim``."""
    if keepdim:
        return shape[:dim] + (1,) + shape[dim + 1:]
    return shape[:dim] + shape[dim + 1:]


def source(dim, keepdim):
    return f"mean({dim}, keepdim=True)" if keepdim else f"mean({dim})"


CASES = [
    (("B", 12), 1, True),
    (("B", 6, 5), 1, False),
    (("B", 6, 5), 2, False),
    (("B", 6, 5), 1, True),
    (("B", 6, 5), 2, True),
    (("B", 3, 4, 5), 1, False),
    (("B", 3, 4, 5), 2, False),
    (("B", 3, 4, 5), 3, False),
    (("B", 3, 4, 5), 1, True),
    (("B", 3, 4, 5), 3, True),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,dim,keepdim", CASES)
def test_matches_torch_on_every_rank_forward_and_backward(device, shape, dim, keepdim):
    plan = resolve(source(dim, keepdim), input_shape=shape, output_shape=reduced(shape, dim, keepdim))
    node = plan.nodes[0]
    assert node.input_shapes["x"] == shape
    assert node.output_shapes["out"] == reduced(shape, dim, keepdim)
    model = build(plan, device=device)
    x = torch.randn(4, *shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = REFERENCE(mirror, dim=dim, keepdim=keepdim)
    assert tuple(actual.shape) == (4, *reduced(shape, dim, keepdim)[1:])
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    torch.testing.assert_close(x.grad, mirror.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_repeated_reductions_pool_an_image_globally(device):
    plan = resolve("mean(2)\nmean(2)", input_shape=("B", 5, 4, 6), output_shape=("B", 5))
    model = build(plan, device=device)
    x = torch.randn(3, 5, 4, 6, device=device)
    torch.testing.assert_close(model(x=x)["output"], REFERENCE(x, dim=(2, 3)))


def test_output_contract_infers_the_surviving_axes_backward():
    plan = resolve("linear()\nmean(1)", input_shape=("B", 5, 8), output_shape=("B", 7))
    assert plan.nodes[0].args["out_features"] == 7
    assert plan.nodes[1].input_shapes["x"] == ("B", 5, 7)

    image = resolve("conv(kernel_size=3, padding=1)\nmean(3)", input_shape=("B", 3, 8, 8), output_shape=("B", 5, 8))
    assert image.nodes[0].args["out_channels"] == 5
    assert image.nodes[1].input_shapes["x"] == ("B", 5, 8, 8)

    kept = resolve("linear()\nmean(1, keepdim=True)", input_shape=("B", 5, 8), output_shape=("B", 1, 7))
    assert kept.nodes[0].args["out_features"] == 7


def test_reduced_extent_is_not_inferable_backward():
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("linear()\nmean(2)", input_shape=("B", 5, 8), output_shape=("B", 5))


def test_rank_two_without_keepdim_is_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*keepdim=True"):
        resolve("mean(1)", input_shape=("B", 12), output_shape=("B", 12))


@pytest.mark.parametrize("dim", [0, -1, 4, 99])
def test_out_of_range_dim_is_an_argument_error(dim):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(f"mean({dim})", input_shape=("B", 3, 4, 5), output_shape=("B", 3, 4))


def test_dim_beyond_the_actual_rank_is_an_argument_error():
    with pytest.raises(HNDLError, match="E_ARGUMENT.*outside rank"):
        resolve("mean(3)", input_shape=("B", 6, 5), output_shape=("B", 6))


def test_conflicting_output_rank_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("mean(1)", input_shape=("B", 6, 5), output_shape=("B", 6, 5))
