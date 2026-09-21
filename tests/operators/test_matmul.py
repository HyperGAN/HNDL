"""Numerics, bidirectional inference, and error codes for ``matmul``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

# (source, input_shape, output_shape, reference) for the rank-3 layouts.
CASES = [
    pytest.param("b = transpose(x, 1, 2)\nmatmul(x, b)", ("B", 6, 4), ("B", 6, 6),
                 lambda x: x @ x.transpose(1, 2), id="square"),
    pytest.param("a, b = split(6, dim=1)\nt = transpose(b, 1, 2)\nmatmul(a, t)",
                 ("B", 9, 4), ("B", 6, 3),
                 lambda x: x[:, :6] @ x[:, 6:].transpose(1, 2), id="rectangular"),
    pytest.param("k = transpose(x, 1, 2)\ne = matmul(x, k)\na = softmax(e, -1)\nmatmul(a, x)",
                 ("B", 5, 3), ("B", 5, 3),
                 lambda x: torch.softmax(x @ x.transpose(1, 2), dim=-1) @ x, id="attention"),
]


@pytest.mark.parametrize("source,input_shape,output_shape,expected", CASES)
def test_matches_torch_on_every_supported_layout(source, input_shape, output_shape, expected, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=13)
    x = torch.randn(3, *input_shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    reference = expected(mirror)
    assert tuple(actual.shape) == (3, *output_shape[1:])
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_the_module_is_torch_bmm_and_carries_no_parameters(device):
    plan = resolve("b = transpose(x, 1, 2)\nmatmul(x, b)", input_shape=("B", 6, 4), output_shape=("B", 6, 6))
    model = build(plan, device=device)
    module = model["n1"]
    assert list(module.parameters()) == [] and list(module.buffers()) == []
    a = torch.randn(2, 6, 4, device=device)
    b = torch.randn(2, 4, 6, device=device)
    torch.testing.assert_close(module(a, b), torch.bmm(a, b))


def test_the_product_is_not_commutative(device):
    contract = dict(input_shape=("B", 8, 4), output_shape=("B", 4, 4))
    plan = resolve("a, b = split(4, dim=1)\nmatmul(a, b)", **contract)
    flipped = resolve("a, b = split(4, dim=1)\nmatmul(b, a)", **contract)
    assert plan.semantic_digest != flipped.semantic_digest
    x = torch.randn(2, 8, 4, device=device)
    left = build(plan, device=device)(x=x)["output"]
    right = build(flipped, device=device)(x=x)["output"]
    torch.testing.assert_close(left, x[:, :4] @ x[:, 4:])
    torch.testing.assert_close(right, x[:, 4:] @ x[:, :4])
    assert not torch.allclose(left, right)


def test_a_flattened_image_attends_and_reshapes_back(device):
    source = ("f = reshape(x, 4)\nt = transpose(f, 1, 2)\ne = matmul(t, f)\n"
              "a = softmax(e, -1)\no = matmul(f, a)\nreshape(o, 4, 3, 3)")
    plan = resolve(source, input_shape=("B", 4, 3, 3), output_shape=("B", 4, 3, 3))
    assert plan.nodes[2].output_shapes["out"] == ("B", 9, 9)
    model = build(plan, device=device)
    x = torch.randn(2, 4, 3, 3, device=device)
    flat = x.reshape(2, 4, 9)
    expected = (flat @ torch.softmax(flat.transpose(1, 2) @ flat, dim=-1)).reshape(2, 4, 3, 3)
    torch.testing.assert_close(model(x=x)["output"], expected)


def test_the_shared_inner_dimension_flows_backward():
    plan = resolve("g = linear()\nt = transpose(x, 1, 2)\nmatmul(g, t)",
                   input_shape=("B", 4, 6), output_shape=("B", 4, 4))
    assert plan.nodes[0].args["out_features"] == 6
    assert plan.nodes[2].input_shapes == {"a": ("B", 4, 6), "b": ("B", 6, 4)}


def test_the_product_shape_flows_forward_into_the_consumer():
    plan = resolve("t = transpose(x, 1, 2)\np = matmul(x, t)\nlinear(p)",
                   input_shape=("B", 5, 3), output_shape=("B", 5, 2))
    assert plan.nodes[1].output_shapes["out"] == ("B", 5, 5)
    assert plan.nodes[2].args["in_features"] == 5


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    # A rank-2 operand: matmul is declared at rank 3 only.
    ("a = linear(4)\nb = linear(x, 4)\nmatmul(a, b)", ("B", 8), ("B", 4), "E_CONSTRAINT"),
    # A rank-4 operand must be reshaped to [B, C, H*W] first.
    ("a = conv(3, kernel_size=1)\nb = conv(x, 3, kernel_size=1)\nmatmul(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    # The inner dimensions disagree: 5 columns against 4 rows.
    ("a = linear(5)\nb = transpose(x, 1, 2)\nmatmul(a, b)", ("B", 6, 4), ("B", 6, 6), "E_CONSTRAINT"),
    # The output contract contradicts the product.
    ("k = transpose(x, 1, 2)\nmatmul(x, k)", ("B", 6, 4), ("B", 6, 5), "E_CONSTRAINT"),
    ("matmul(x)", ("B", 6, 4), ("B", 6, 4), "E_BINDING"),
    ("matmul()", ("B", 6, 4), ("B", 6, 4), "E_BINDING"),
])
def test_invalid_graphs_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)
