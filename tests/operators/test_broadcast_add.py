"""Numerics, gradients, determinism, inference, and error codes for ``broadcast_add``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

# (source, input_shape, output_shape, reference) covering ranks 2, 3, and 4.
CASES = [
    pytest.param("a = linear(4)\nb = mean(x, 1, keepdim=True)\nbroadcast_add(a, b)", ("B", 6), ("B", 4),
                 lambda model, x: model["n0"](x) + x.mean(dim=1, keepdim=True), id="rank2-scalar-bias"),
    pytest.param("h = linear(8)\nsummary = mean(h, 1, keepdim=True)\nbroadcast_add(h, summary)",
                 ("B", 6, 4), ("B", 6, 8),
                 lambda model, x: model["n0"](x) + model["n0"](x).mean(dim=1, keepdim=True),
                 id="rank3-summary-token"),
    pytest.param("saved = x\nglobal_avg_pool()\nlinear(3)\nbias = reshape(3, 1, 1)\nbroadcast_add(saved, bias)",
                 ("B", 3, 8, 8), ("B", 3, 8, 8),
                 lambda model, x: x + model["n1"](x.mean(dim=(2, 3))).reshape(x.shape[0], 3, 1, 1),
                 id="rank4-channel-bias"),
]


@pytest.mark.parametrize("source,input_shape,output_shape,expected", CASES)
def test_matches_torch_on_every_supported_rank(source, input_shape, output_shape, expected, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=11)
    x = torch.randn(3, *input_shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    reference = expected(model, mirror)
    assert tuple(actual.shape) == (3, *output_shape[1:])
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_broadcast_matches_torch_add_including_both_gradients(device):
    """The reduced operand's gradient is the sum over the broadcast axes."""
    module = build(resolve("saved = x\nglobal_avg_pool()\nlinear(3)\nbias = reshape(3, 1, 1)\n"
                           "broadcast_add(saved, bias)",
                           input_shape=("B", 3, 8, 8), output_shape=("B", 3, 8, 8)),
                   device=device)["n3"]
    a = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
    b = torch.randn(2, 3, 1, 1, device=device, requires_grad=True)
    mirrors = [t.detach().clone().requires_grad_() for t in (a, b)]
    out = module(a, b)
    expected = torch.add(*mirrors)
    assert tuple(out.shape) == (2, 3, 8, 8)
    torch.testing.assert_close(out, expected)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad.clone())
    torch.testing.assert_close(a.grad, grad)
    torch.testing.assert_close(b.grad, grad.sum(dim=(2, 3), keepdim=True))
    torch.testing.assert_close(a.grad, mirrors[0].grad)
    torch.testing.assert_close(b.grad, mirrors[1].grad)


@pytest.mark.parametrize("shapes", [((2, 3, 8, 8), (2, 3, 1, 1)), ((2, 1, 8, 8), (2, 4, 1, 1)),
                                    ((2, 6, 5), (2, 1, 5)), ((2, 6, 5), (2, 6, 1))])
def test_symmetric_broadcasting_in_either_operand(shapes, device):
    module = build(resolve("a, b = split(3)\nbroadcast_add(a, b)",
                           input_shape=("B", 6), output_shape=("B", 3)), device=device)["n1"]
    a = torch.randn(*shapes[0], device=device)
    b = torch.randn(*shapes[1], device=device)
    torch.testing.assert_close(module(a, b), a + b)
    torch.testing.assert_close(module(b, a), b + a)


def test_backward_is_deterministic_under_the_deterministic_flag(device):
    """``sum`` reduction of the broadcast axes uses no atomics, so it is bit-exact."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        plan = resolve("saved = x\nglobal_avg_pool()\nlinear(3)\nbias = reshape(3, 1, 1)\n"
                       "broadcast_add(saved, bias)",
                       input_shape=("B", 3, 16, 16), output_shape=("B", 3, 16, 16))
        model = build(plan, device=device, initialization_seed=4)
        x = torch.randn(4, 3, 16, 16, device=device)
        grads = []
        for _ in range(3):
            a = x.clone().requires_grad_()
            b = torch.randn(4, 3, 1, 1, device=device, generator=torch.Generator(device=device).manual_seed(0))
            b.requires_grad_()
            out = model["n3"](a, b)
            out.mul(torch.linspace(-1, 1, out.numel(), device=device).reshape(out.shape)).sum().backward()
            grads.append((a.grad.clone(), b.grad.clone()))
        for grad_a, grad_b in grads[1:]:
            assert torch.equal(grad_a, grads[0][0])
            assert torch.equal(grad_b, grads[0][1])
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.are_deterministic_algorithms_enabled() == previous


def test_second_derivatives_flow_through_both_operands(device):
    module = build(resolve("a, b = split(3)\nbroadcast_add(a, b)",
                           input_shape=("B", 6), output_shape=("B", 3)), device=device)["n1"]
    a = torch.randn(2, 4, 5, 5, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, 4, 1, 1, device=device, dtype=torch.float64, requires_grad=True)
    out = module(a, b)
    first = torch.autograd.grad(out.pow(3).sum(), (a, b), create_graph=True)
    assert first[1].shape == b.shape
    second = torch.autograd.grad(first[0].square().sum() + first[1].square().sum(), (a, b))
    assert all(grad.isfinite().all() for grad in second)
    torch.autograd.gradcheck(lambda p, q: module(p, q), (a, b))
    torch.autograd.gradgradcheck(lambda p, q: module(p, q), (a, b))


def test_a_size_one_operand_resolves_the_other_backward_from_the_contract():
    plan = resolve("gate = mean(2, keepdim=True)\nh = linear(x)\nbroadcast_add(h, gate)",
                   input_shape=("B", 6, 4), output_shape=("B", 6, 8))
    assert plan.nodes[1].args["out_features"] == 8
    assert plan.nodes[2].input_shapes == {"a": ("B", 6, 8), "b": ("B", 6, 1)}
    assert plan.nodes[2].output_shapes["out"] == ("B", 6, 8)


def test_widths_flow_forward_into_the_consumer():
    plan = resolve("h = linear(6)\nbias = mean(h, 1, keepdim=True)\np = broadcast_add(h, bias)\nlinear(p, 2)",
                   input_shape=("B", 5, 3), output_shape=("B", 5, 2))
    assert plan.nodes[2].output_shapes["out"] == ("B", 5, 6)
    assert plan.nodes[3].args["in_features"] == 6


def test_a_unit_output_extent_forces_both_operands_to_one():
    plan = resolve("a = mean(2, keepdim=True)\nb = mean(x, 2, keepdim=True)\nbroadcast_add(a, b)",
                   input_shape=("B", 3, 5), output_shape=("B", 3, 1))
    assert plan.nodes[2].input_shapes == {"a": ("B", 3, 1), "b": ("B", 3, 1)}


def test_plain_add_still_refuses_to_broadcast():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("saved = x\nbias = mean(2, keepdim=True)\nadd(saved, bias)",
                input_shape=("B", 3, 5), output_shape=("B", 3, 5))


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    # Incompatible extents: neither operand is 1 on the feature axis.
    ("a = linear(4)\nb = linear(x, 6)\nbroadcast_add(a, b)", ("B", 8), ("B", 4), "E_CONSTRAINT"),
    # Incompatible spatial extents.
    ("a = conv(3, kernel_size=3, padding=1)\nb = conv(x, 3, kernel_size=3)\nbroadcast_add(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    # Rank mismatch: [B, C, H, W] against [B, C]; ranks are never padded.
    ("a = conv(3, kernel_size=3, padding=1)\nb = global_avg_pool(x)\nbroadcast_add(a, b)",
     ("B", 3, 8, 8), ("B", 3, 8, 8), "E_CONSTRAINT"),
    # An operand wider than the declared output cannot broadcast into it.
    ("a = linear(6)\nb = mean(x, 1, keepdim=True)\nbroadcast_add(a, b)", ("B", 8), ("B", 4), "E_CONSTRAINT"),
    ("linear(4)\nbroadcast_add(x)", ("B", 8), ("B", 4), "E_BINDING"),
    ("linear(4)\nbroadcast_add()", ("B", 8), ("B", 4), "E_BINDING"),
])
def test_invalid_graphs_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_the_incompatible_extent_message_names_the_axis():
    with pytest.raises(HNDLError, match="axis 3"):
        resolve("a = conv(3, kernel_size=3, padding=1)\nb = conv(x, 3, kernel_size=(1, 3))\n"
                "broadcast_add(a, b)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 8, 8))
