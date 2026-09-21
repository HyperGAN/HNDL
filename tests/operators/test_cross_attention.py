"""Numerics, inference, and error codes for the ``cross_attention`` operator."""

import math

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.operators.cross_attention import CrossAttention
from hndl.torch import build

from .conftest import DEVICES


def handwritten(module, x, context):
    """An independent attention implementation written with einsum."""
    heads, head_dim = module.heads, module.head_dim
    def project(layer, tensor):
        projected = F.linear(tensor, layer.weight, layer.bias)
        return projected.unflatten(-1, (heads, head_dim))
    q, k, v = project(module.q_proj, x), project(module.k_proj, context), project(module.v_proj, context)
    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
    attended = torch.einsum("bhts,bshd->bthd", scores.softmax(dim=-1), v)
    return F.linear(attended.flatten(-2), module.o_proj.weight, module.o_proj.bias)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("positions,sources,width,context_width,heads,bias", [
    (3, 5, 8, 8, 2, True),
    (2, 7, 12, 4, 4, True),
    (6, 3, 6, 10, 3, False),
    (4, 4, 16, 3, 1, True),
])
def test_matches_handwritten_attention_and_gradients(device, positions, sources, width, context_width, heads, bias):
    torch.manual_seed(positions * 31 + sources)
    with torch.device(device):
        module = CrossAttention(heads=heads, bias=bias, dropout=0.0, D=width, D_ctx=context_width)
    x = torch.randn(2, positions, width, device=device, requires_grad=True)
    context = torch.randn(2, sources, context_width, device=device, requires_grad=True)
    mirrors = [tensor.detach().clone().requires_grad_() for tensor in (x, context)]

    actual = module(x, context)
    expected = handwritten(module, *mirrors)
    assert actual.shape == (2, positions, width)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for tensor, mirror in zip((x, context), mirrors):
        torch.testing.assert_close(tensor.grad, mirror.grad, atol=2e-6, rtol=2e-6)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


@pytest.mark.parametrize("device", DEVICES)
def test_a_single_query_position_attends_over_the_whole_context(device):
    torch.manual_seed(7)
    with torch.device(device):
        module = CrossAttention(heads=4, bias=True, dropout=0.0, D=12, D_ctx=5)
    x, context = torch.randn(2, 1, 12, device=device), torch.randn(2, 7, 5, device=device)
    torch.testing.assert_close(module(x, context), handwritten(module, x, context), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("sources,width,heads", [(1, 6, 3), (4, 4, 4)])
def test_single_position_contexts_and_one_feature_per_head_match_the_reference(sources, width, heads):
    # CPU only: a context of length one, or one feature per head, turns an
    # attention matmul into an outer product, which this CUDA build services
    # with a Triton kernel the test machine cannot compile. The arithmetic
    # itself is device independent.
    torch.manual_seed(5)
    module = CrossAttention(heads=heads, bias=False, dropout=0.0, D=width, D_ctx=6)
    x, context = torch.randn(2, 5, width), torch.randn(2, sources, 6)
    torch.testing.assert_close(module(x, context), handwritten(module, x, context), atol=2e-6, rtol=2e-6)


def test_gradcheck_in_double_precision():
    torch.manual_seed(3)
    module = CrossAttention(heads=2, bias=True, dropout=0.0, D=4, D_ctx=3).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    context = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x, context))


@pytest.mark.parametrize("device", DEVICES)
def test_attention_weights_are_a_convex_combination_of_context_values(device):
    """With identity value and output projections the result lies in the convex hull of v."""
    torch.manual_seed(0)
    with torch.device(device):
        module = CrossAttention(heads=1, bias=False, dropout=0.0, D=4, D_ctx=4)
    with torch.no_grad():
        module.v_proj.weight.copy_(torch.eye(4, device=device))
        module.o_proj.weight.copy_(torch.eye(4, device=device))
    context = torch.randn(2, 5, 4, device=device)
    out = module(torch.randn(2, 3, 4, device=device), context)
    lower, upper = context.amin(dim=1, keepdim=True), context.amax(dim=1, keepdim=True)
    assert bool(((out >= lower - 1e-5) & (out <= upper + 1e-5)).all())


def test_plan_carries_distinct_query_and_context_widths():
    plan = resolve("q, kv = split(4)\nc = linear(kv, 16)\ncross_attention(q, c, 2)",
                   input_shape=("B", 12, 8), output_shape=("B", 4, 8))
    node = plan.nodes[-1]
    assert node.input_shapes == {"x": ("B", 4, 8), "context": ("B", 8, 16)}
    assert node.output_shapes["out"] == ("B", 4, 8)
    assert node.args == {"heads": 2, "bias": True, "dropout": 0.0}
    model = build(plan, device="cpu")
    module = model[node.id]
    assert module.q_proj.weight.shape == (8, 8) and module.k_proj.weight.shape == (8, 16)
    assert module.v_proj.weight.shape == (8, 16) and module.o_proj.weight.shape == (8, 8)
    out = model(x=torch.randn(2, 12, 8))["output"]
    assert out.shape == (2, 4, 8)


def test_query_width_is_inferred_backwards_from_the_output_contract():
    plan = resolve("q, kv = split(4)\nh = linear(q)\ncross_attention(h, kv, 4)",
                   input_shape=("B", 12, 8), output_shape=("B", 4, 16))
    assert plan.nodes[1].args["out_features"] == 16
    assert plan.nodes[-1].input_shapes["x"] == ("B", 4, 16)
    assert plan.nodes[-1].output_shapes["out"] == ("B", 4, 16)
    forward = resolve("q, kv = split(4)\ncross_attention(q, kv, 2)\nlinear()",
                      input_shape=("B", 12, 8), output_shape=("B", 4, 3))
    assert forward.nodes[1].output_shapes["out"] == ("B", 4, 8)
    assert forward.nodes[2].args["in_features"] == 8


@pytest.mark.parametrize("source,code", [
    ("q, kv = split(4)\ncross_attention(q, kv, 3)", "E_CONSTRAINT"),
    ("q, kv = split(4)\nc = linear(kv, 6)\ncross_attention(q, c, 5)", "E_CONSTRAINT"),
    ("q, kv = split(4)\ncross_attention(q, kv, 0)", "E_ARGUMENT"),
    ("q, kv = split(4)\ncross_attention(q, kv, 2, dropout=1.0)", "E_ARGUMENT"),
    ("q, kv = split(4)\ncross_attention(q, kv)", "E_ARGUMENT"),
    ("cross_attention(2)", "E_BINDING"),
    ("q, kv = split(4)\ncross_attention(q, 2)", "E_BINDING"),
])
def test_invalid_configurations_report_their_code(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 12, 8), output_shape=("B", 4, 8))


def test_unresolvable_context_width_is_ambiguous_rather_than_guessed():
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("q, kv = split(4)\nc = linear(kv)\ncross_attention(q, c, 2)",
                input_shape=("B", 12, 8), output_shape=("B", 4, 8))


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_only_perturbs_training_and_eval_matches_the_reference(device):
    plan = resolve("q, kv = split(4)\ncross_attention(q, kv, 2, dropout=0.5)",
                   input_shape=("B", 12, 8), output_shape=("B", 4, 8))
    assert plan.nodes[-1].args["dropout"] == 0.5
    model = build(plan, device=device, initialization_seed=11)
    x = torch.randn(2, 12, 8, device=device)

    model.train()
    torch.manual_seed(0)
    first = model(x=x)["output"]
    torch.manual_seed(1)
    second = model(x=x)["output"]
    assert not torch.allclose(first, second)

    model.eval()
    clean = model(x=x)["output"]
    torch.testing.assert_close(model(x=x)["output"], clean)
    query, context = x[:, :4], x[:, 4:]
    torch.testing.assert_close(clean, handwritten(model[plan.nodes[-1].id], query, context),
                               atol=2e-6, rtol=2e-6)
