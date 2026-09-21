"""Numerics, bidirectional inference, and error codes for ``mul``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

# (source, input_shape, output_shape, reference) for ranks 2, 3, and 4.
CASES = [
    pytest.param("a, b = split(3)\nmul(a, b)", ("B", 6), ("B", 3),
                 lambda x: x[:, :3] * x[:, 3:], id="rank2"),
    pytest.param("a, b = split(4, dim=2)\nmul(a, b)", ("B", 5, 8), ("B", 5, 4),
                 lambda x: x[:, :, :4] * x[:, :, 4:], id="rank3"),
    pytest.param("a, b = split(2, dim=1)\nmul(a, b)", ("B", 4, 6, 6), ("B", 2, 6, 6),
                 lambda x: x[:, :2] * x[:, 2:], id="rank4"),
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


def test_product_is_commutative_but_the_plan_records_the_written_order(device):
    plan = resolve("a, b = split(3)\nmul(a, b)", input_shape=("B", 6), output_shape=("B", 3))
    flipped = resolve("a, b = split(3)\nmul(b, a)", input_shape=("B", 6), output_shape=("B", 3))
    x = torch.randn(4, 6, device=device)
    torch.testing.assert_close(build(plan, device=device)(x=x)["output"],
                               build(flipped, device=device)(x=x)["output"])
    assert plan.semantic_digest != flipped.semantic_digest


def test_a_gate_branch_matches_the_handwritten_equivalent(device):
    source = "h = linear(4)\ng = linear(x, 4)\ng = tanh(g)\nmul(h, g)"
    plan = resolve(source, input_shape=("B", 8), output_shape=("B", 4))
    model = build(plan, device=device, initialization_seed=7)
    x = torch.randn(2, 8, device=device)
    expected = model["n0"](x) * torch.tanh(model["n1"](x))
    torch.testing.assert_close(model(x=x)["output"], expected)


def test_widths_flow_backward_from_the_output_contract():
    plan = resolve("saved = linear(6)\nh = linear(saved)\nmul(h, saved)",
                   input_shape=("B", 3), output_shape=("B", 6))
    assert plan.nodes[1].args["out_features"] == 6
    assert plan.nodes[2].input_shapes == {"a": ("B", 6), "b": ("B", 6)}


def test_widths_flow_forward_into_the_consumer():
    plan = resolve("saved = linear(6)\nh = linear(saved, 6)\np = mul(h, saved)\nlinear(p)",
                   input_shape=("B", 3), output_shape=("B", 2))
    assert plan.nodes[2].output_shapes["out"] == ("B", 6)
    assert plan.nodes[3].args["in_features"] == 6


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("a = linear(4)\nb = linear(x, 6)\nmul(a, b)", ("B", 8), ("B", 4), "E_CONSTRAINT"),
    ("a = conv(3, kernel_size=3, padding=1)\nb = flatten(x)\nmul(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    ("a = conv(3, kernel_size=3, padding=1)\nb = conv(x, 3, kernel_size=3)\nmul(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    ("linear(4)\nmul(x)", ("B", 8), ("B", 4), "E_BINDING"),
    ("linear(4)\nmul()", ("B", 8), ("B", 4), "E_BINDING"),
])
def test_invalid_graphs_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)
