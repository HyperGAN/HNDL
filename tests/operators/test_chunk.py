"""Numerics, equal-split inference, and error codes for ``chunk``."""

import pytest
import torch

from hndl import HNDLError, ResolvedPlan, Registry, resolve, resolve_callable
from hndl.torch import build

# (source, input_shape, output_shape, reference) for the non-batch axes.
CASES = [
    pytest.param("a, b = chunk(2)\nconcat(b, a)", ("B", 8), ("B", 8),
                 lambda x: torch.cat((x[:, 4:], x[:, :4]), dim=1), id="rank2"),
    pytest.param("a, b, c = chunk(3, dim=2)\nconcat(a, c, b, axis=2)", ("B", 4, 9), ("B", 4, 9),
                 lambda x: torch.cat((x[:, :, :3], x[:, :, 6:], x[:, :, 3:6]), dim=2), id="rank3"),
    pytest.param("a, b = chunk(2, dim=1)\nconcat(b, a, axis=1)", ("B", 6, 5, 5), ("B", 6, 5, 5),
                 lambda x: torch.cat((x[:, 3:], x[:, :3]), dim=1), id="rank4"),
]


@pytest.mark.parametrize("source,input_shape,output_shape,expected", CASES)
def test_equal_sections_match_torch_on_every_supported_rank(source, input_shape, output_shape, expected, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device)
    x = torch.randn(3, *input_shape[1:], device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual, reference = model(x=x)["output"], expected(mirror)
    assert tuple(actual.shape) == (3, *output_shape[1:])
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_sections_are_views_that_keep_autograd(device):
    plan = resolve("a, b = chunk(2)\nlinear(a, 3)", input_shape=("B", 8), output_shape=("B", 3))
    module = build(plan, device=device)["n0"]
    x = torch.randn(2, 8, device=device, requires_grad=True)
    sections = module(x)
    assert len(sections) == 2 and all(tuple(part.shape) == (2, 4) for part in sections)
    torch.testing.assert_close(torch.cat(sections, dim=1), x)
    sections[1].square().sum().backward()
    assert torch.equal(x.grad[:, :4], torch.zeros_like(x.grad[:, :4]))
    assert x.grad[:, 4:].abs().sum() > 0


def test_the_input_extent_is_inferred_from_a_section():
    plan = resolve("a, b = chunk(2)\np = linear(a, 3)\nq = linear(b, 3)\nadd(p, q)",
                   input_shape=("B", 10), output_shape=("B", 3))
    assert plan.nodes[0].output_shapes == {"out0": ("B", 5), "out1": ("B", 5)}


def test_chunks_of_one_returns_a_one_tuple_and_clears_current():
    plan = resolve("a, = chunk(1)\nlinear(a, 5)", input_shape=("B", 4), output_shape=("B", 5))
    assert plan.nodes[0].outputs == ("out0",)
    assert plan.nodes[0].output_shapes == {"out0": ("B", 4)}
    with pytest.raises(HNDLError, match="E_CURRENT"):
        resolve("chunk(1)\nlinear(5)", input_shape=("B", 4), output_shape=("B", 5))


def test_the_ports_and_the_argument_round_trip_together():
    registry = Registry.builtins()
    plan = resolve("a, b, c = chunk(3)\nconcat(c, b, a)", input_shape=("B", 9), output_shape=("B", 9),
                   registry=registry)
    restored = ResolvedPlan.from_json(plan.to_json(), registry=registry)
    assert restored.semantic_digest == plan.semantic_digest
    assert restored.nodes[0].outputs == ("out0", "out1", "out2")
    tampered = plan.to_json().replace('"chunks":3', '"chunks":2', 1)
    with pytest.raises(HNDLError, match="E_INTEGRITY|E_BINDING"):
        ResolvedPlan.from_json(tampered, registry=registry)


def test_both_frontends_unpack_the_same_sections():
    registry = Registry.builtins()
    kwargs = {"input_shape": ("B", 8), "output_shape": ("B", 8), "registry": registry}
    plan = resolve("a, b = chunk(2)\nconcat(b, a)", **kwargs)

    def author(x):
        a, b = registry.ops.chunk(x, 2, name="n0")
        return registry.ops.concat(b, a, name="n1")

    assert resolve_callable(author, **kwargs).semantic_digest == plan.semantic_digest


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("a, b = chunk(2, dim=1)\nconcat(a, b)", ("B", 5), ("B", 5), "E_CONSTRAINT"),
    ("a, b, c = chunk(3)\nconcat(a, b, c)", ("B", 8), ("B", 8), "E_CONSTRAINT"),
    ("a, b = chunk(2, dim=3)\nconcat(a, b, axis=3)", ("B", 4, 8), ("B", 4, 8), "E_ARGUMENT"),
    ("a, b = chunk(2, dim=-1)", ("B", 8), ("B", 4), "E_ARGUMENT"),
    ("a, b = chunk(0)", ("B", 8), ("B", 8), "E_ARGUMENT"),
    ("a, b = chunk(2)\nconcat(a, b)", ("B", 3), ("B", 3), "E_CONSTRAINT"),
    ("a, b, c = chunk(2)", ("B", 8), ("B", 4), "E_OUTPUT_ARITY"),
])
def test_invalid_graphs_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)
