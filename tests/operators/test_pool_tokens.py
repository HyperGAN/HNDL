"""Numerics, bidirectional inference, and error codes for ``pool_tokens``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

MODES = ["mean", "first", "last", "max"]


def expected(mode, x):
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "first":
        return x[:, 0]
    if mode == "last":
        return x[:, -1]
    return x.amax(dim=1)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch,tokens,width", [(1, 1, 3), (2, 7, 5), (4, 3, 16)])
def test_pooling_matches_the_torch_reference_with_gradients(mode, batch, tokens, width, device):
    plan = resolve(f'pool_tokens("{mode}")', input_shape=("B", tokens, width), output_shape=("B", width))
    assert plan.nodes[0].args["mode"] == mode
    model = build(plan, device=device, initialization_seed=1)
    x = torch.randn(batch, tokens, width, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    reference = expected(mode, mirror)
    assert actual.shape == (batch, width)
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_max_selects_the_elementwise_maximum_across_positions(device):
    plan = resolve('pool_tokens("max")', input_shape=("B", 3, 2), output_shape=("B", 2))
    model = build(plan, device=device)
    x = torch.tensor([[[1.0, -5.0], [-2.0, 4.0], [0.5, 4.0]]], device=device)
    torch.testing.assert_close(model(x=x)["output"], torch.tensor([[1.0, 4.0]], device=device))


def test_default_mode_is_mean_and_is_recorded_in_the_plan():
    plan = resolve("pool_tokens()", input_shape=("B", 4, 6), output_shape=("B", 6))
    assert plan.nodes[0].args["mode"] == "mean"
    assert "mode='mean'" in repr(build(plan, device="cpu")["n0"])


def test_width_flows_backward_from_the_pooled_output():
    plan = resolve("linear()\npool_tokens()", input_shape=("B", 4, 8), output_shape=("B", 16))
    assert plan.nodes[0].args["out_features"] == 16
    assert plan.nodes[1].input_shapes["x"] == ("B", 4, 16)
    assert plan.nodes[1].output_shapes["out"] == ("B", 16)


def test_width_flows_forward_into_the_consumer():
    plan = resolve('pool_tokens("last")\nlinear()', input_shape=("B", 5, 12), output_shape=("B", 3))
    assert plan.nodes[1].args["in_features"] == 12
    assert plan.nodes[0].output_shapes["out"] == ("B", 12)


def test_the_token_count_is_solved_from_the_element_count_when_pooling_hides_it():
    plan = resolve("reshape()\npool_tokens()", input_shape=("B", 24), output_shape=("B", 6))
    assert plan.nodes[0].args["shape"] == (4, 6)
    assert plan.nodes[1].input_shapes["x"] == ("B", 4, 6)


@pytest.mark.parametrize("source,code", [
    ('pool_tokens("median")', "E_ARGUMENT"),
    ("pool_tokens(4)", "E_ARGUMENT"),
])
def test_invalid_modes_are_rejected(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 4, 6), output_shape=("B", 6))


@pytest.mark.parametrize("input_shape,output_shape", [
    (("B", 6), ("B", 6)),
    (("B", 3, 4, 4), ("B", 3)),
])
def test_non_sequence_inputs_are_a_shape_contradiction(input_shape, output_shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("pool_tokens()", input_shape=input_shape, output_shape=output_shape)


def test_the_pooled_width_must_match_the_incoming_width():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("pool_tokens()", input_shape=("B", 4, 6), output_shape=("B", 5))
