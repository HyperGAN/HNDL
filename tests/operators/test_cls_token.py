"""Numerics, inference, and error codes specific to ``cls_token``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build, network


def test_the_token_is_prepended_and_the_sequence_is_untouched(device):
    model = network("cls_token()", input_shape=("B", 4, 5), output_shape=("B", 5, 5),
                    device=device, initialization_seed=9)
    token = model["n0"].token
    assert tuple(token.shape) == (1, 1, 5)
    x = torch.randn(3, 4, 5, device=device)
    output = model(x)
    assert output.shape == (3, 5, 5)
    torch.testing.assert_close(output[:, 1:], x)
    torch.testing.assert_close(output[:, 0], token[0, 0].expand(3, 5))
    torch.testing.assert_close(output, torch.cat((token.expand(3, 1, 5), x), dim=1))


def test_the_gradient_reaches_the_token_and_the_input(device):
    model = network("cls_token()", input_shape=("B", 3, 4), output_shape=("B", 4, 4),
                    device=device, initialization_seed=4)
    x = torch.randn(2, 3, 4, device=device, requires_grad=True)
    output = model(x)
    (output * torch.arange(1, output.numel() + 1, device=device).reshape(output.shape)).sum().backward()
    token_grad = model["n0"].token.grad
    assert token_grad is not None and tuple(token_grad.shape) == (1, 1, 4)
    assert (token_grad != 0).all()
    assert x.grad is not None and (x.grad != 0).all()


def test_the_position_count_is_solved_in_both_directions():
    forward = resolve("cls_token()\nlinear()", input_shape=("B", 6, 4), output_shape=("B", 7, 3))
    assert forward.nodes[0].output_shapes["out"] == ("B", 7, 4)

    backward = resolve("reshape(4, 3)\ncls_token()", input_shape=("B", 12), output_shape=("B", 5, 3))
    assert backward.nodes[0].args["shape"] == (4, 3)
    assert backward.nodes[1].input_shapes["x"] == ("B", 4, 3)

    inferred = resolve("reshape()\ncls_token()", input_shape=("B", 4, 3), output_shape=("B", 5, 3))
    assert inferred.nodes[1].output_shapes["out"] == ("B", 5, 3)


def test_a_contract_of_one_position_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*at least two positions"):
        resolve("cls_token()", input_shape=("B", 1, 4), output_shape=("B", 1, 4))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("cls_token()", input_shape=("B", 4, 3), output_shape=("B", 4, 3))


def test_rank_two_and_rank_four_inputs_are_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("cls_token()", input_shape=("B", 12), output_shape=("B", 13))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("cls_token()", input_shape=("B", 2, 3, 4), output_shape=("B", 3, 3, 4))


def test_the_token_is_counted_by_a_following_position_table():
    plan = resolve("cls_token()\npos_embed(5)", input_shape=("B", 4, 3), output_shape=("B", 5, 3))
    assert plan.nodes[1].input_shapes["x"] == ("B", 5, 3)
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*max_len"):
        resolve("cls_token()\npos_embed(4)", input_shape=("B", 4, 3), output_shape=("B", 5, 3))


def test_parameter_count_is_the_feature_width(device):
    model = network("cls_token()", input_shape=("B", 2, 16), output_shape=("B", 3, 16), device=device)
    assert sum(p.numel() for p in model.parameters()) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_build_on_cuda(dtype):
    plan = resolve("cls_token()\nlinear(4)", input_shape=("B", 6, 8), output_shape=("B", 7, 4), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=3)
    torch_dtype = getattr(torch, dtype)
    assert model["n0"].token.dtype == torch_dtype
    x = torch.randn(2, 6, 8, device="cuda:0", dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (2, 7, 4)
    output.float().square().mean().backward()
    assert model["n0"].token.grad.isfinite().all()
