"""Passthrough semantics, inference, and argument errors for ``identity``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

SHAPES = [("B", 16), ("B", 4, 8), ("B", 2, 5, 5)]


@pytest.mark.parametrize("shape", SHAPES)
def test_returns_the_input_unchanged_and_passes_the_gradient_through(shape, device):
    plan = resolve("identity()", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x = torch.randn(3, *shape[1:], device=device, requires_grad=True)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, x)
    assert out.data_ptr() == x.data_ptr()
    out.mul(2.0).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


def test_has_no_parameters(device):
    plan = resolve("identity()", input_shape=("B", 8), output_shape=("B", 8))
    model = build(plan, device=device)
    assert list(model[plan.nodes[0].id].parameters()) == []


def test_shapes_flow_backward_through_the_passthrough():
    plan = resolve("linear()\nidentity()", input_shape=("B", 32), output_shape=("B", 7))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7)
    assert plan.nodes[1].input_shapes["x"] == ("B", 7)


def test_shapes_flow_forward_through_the_passthrough():
    plan = resolve("identity()\nlinear()", input_shape=("B", 3, 6), output_shape=("B", 3, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6)
    assert plan.nodes[0].input_shapes["x"] == ("B", 3, 6)


def test_named_passthrough_feeds_a_residual_branch(device):
    source = 'h = identity(name="hidden")\nlinear(8, name="wide")\nrelu(name="act")\ny = linear(4, name="narrow")\nadd(y, h)'
    plan = resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    assert plan.nodes[0].id == "hidden"
    assert plan.nodes[-1].inputs["b"] == "node:hidden/out"
    model = build(plan, device=device, initialization_seed=1)
    x = torch.randn(3, 4, device=device)
    out = model(x=x)["output"]
    branch = model["narrow"](model["act"](model["wide"](x)))
    torch.testing.assert_close(out, branch + x)


@pytest.mark.parametrize("source", ["identity(1)", "identity(alpha=1.0)"])
def test_unexpected_arguments_report_e_argument(source):
    with pytest.raises(HNDLError) as excinfo:
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"
