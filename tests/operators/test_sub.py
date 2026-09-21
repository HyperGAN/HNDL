"""Numerics, bidirectional inference, and error codes for ``sub``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

# (source, input_shape, output_shape, reference) for ranks 2, 3, and 4.
CASES = [
    pytest.param("a, b = split(3)\nsub(a, b)", ("B", 6), ("B", 3),
                 lambda x: x[:, :3] - x[:, 3:], id="rank2"),
    pytest.param("a, b = split(4, dim=2)\nsub(a, b)", ("B", 5, 8), ("B", 5, 4),
                 lambda x: x[:, :, :4] - x[:, :, 4:], id="rank3"),
    pytest.param("a, b = split(2, dim=1)\nsub(a, b)", ("B", 4, 6, 6), ("B", 2, 6, 6),
                 lambda x: x[:, :2] - x[:, 2:], id="rank4"),
]


@pytest.mark.parametrize("source,input_shape,output_shape,expected", CASES)
def test_matches_torch_on_every_supported_rank(source, input_shape, output_shape, expected, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=11)
    x = torch.randn(3, *input_shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    reference = expected(mirror)
    assert tuple(actual.shape) == (3, *output_shape[1:])
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_subtraction_is_ordered_and_anti_commutative(device):
    plan = resolve("a, b = split(3)\nsub(a, b)", input_shape=("B", 6), output_shape=("B", 3))
    flipped = resolve("a, b = split(3)\nsub(b, a)", input_shape=("B", 6), output_shape=("B", 3))
    x = torch.randn(4, 6, device=device)
    left = build(plan, device=device)(x=x)["output"]
    right = build(flipped, device=device)(x=x)["output"]
    torch.testing.assert_close(left, -right)
    assert plan.semantic_digest != flipped.semantic_digest


def test_widths_flow_backward_from_the_output_contract():
    plan = resolve("saved = linear(6)\nh = linear(saved)\nsub(h, saved)",
                   input_shape=("B", 3), output_shape=("B", 6))
    assert plan.nodes[1].args["out_features"] == 6
    assert plan.nodes[2].input_shapes == {"a": ("B", 6), "b": ("B", 6)}


def test_widths_flow_forward_into_the_consumer():
    plan = resolve("saved = linear(6)\nh = linear(saved, 6)\nd = sub(h, saved)\nlinear(d)",
                   input_shape=("B", 3), output_shape=("B", 2))
    assert plan.nodes[2].output_shapes["out"] == ("B", 6)
    assert plan.nodes[3].args["in_features"] == 6


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("a = linear(4)\nb = linear(x, 6)\nsub(a, b)", ("B", 8), ("B", 4), "E_CONSTRAINT"),
    ("a = conv(3, kernel_size=3, padding=1)\nb = flatten(x)\nsub(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    ("a = conv(3, kernel_size=3, padding=1)\nb = conv(x, 3, kernel_size=3)\nsub(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    ("linear(4)\nsub(x)", ("B", 8), ("B", 4), "E_BINDING"),
    ("linear(4)\nsub()", ("B", 8), ("B", 4), "E_BINDING"),
])
def test_invalid_graphs_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)
