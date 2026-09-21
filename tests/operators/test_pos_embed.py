"""Numerics, inference, and error codes specific to ``pos_embed``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build, network


def test_matches_the_explicit_broadcast_sum(device):
    model = network("pos_embed(16)", input_shape=("B", 6, 5), output_shape=("B", 6, 5),
                    device=device, initialization_seed=7)
    table = model["n0"].weight
    assert tuple(table.shape) == (16, 5)
    x = torch.randn(3, 6, 5, device=device)
    torch.testing.assert_close(model(x), x + table[:6].unsqueeze(0))


def test_the_same_vector_is_added_to_every_example(device):
    model = network("pos_embed(8)", input_shape=("B", 4, 3), output_shape=("B", 4, 3), device=device)
    x = torch.zeros(5, 4, 3, device=device)
    output = model(x)
    for row in range(1, 5):
        torch.testing.assert_close(output[0], output[row])
    torch.testing.assert_close(output[0], model["n0"].weight[:4])


def test_only_the_used_rows_receive_a_gradient(device):
    model = network("pos_embed(10)", input_shape=("B", 4, 3), output_shape=("B", 4, 3),
                    device=device, initialization_seed=5)
    model(torch.randn(2, 4, 3, device=device)).square().sum().backward()
    grad = model["n0"].weight.grad
    assert (grad[:4] != 0).any()
    assert torch.equal(grad[4:], torch.zeros_like(grad[4:]))


def test_the_table_is_initialized_with_a_small_normal_spread():
    model = network("pos_embed(512)", input_shape=("B", 4, 64), output_shape=("B", 4, 64),
                    device="cpu", initialization_seed=1)
    weight = model["n0"].weight.detach()
    assert 0.01 < weight.std().item() < 0.03
    assert abs(weight.mean().item()) < 0.005


def test_a_sequence_longer_than_the_table_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*max_len"):
        resolve("pos_embed(4)", input_shape=("B", 5, 3), output_shape=("B", 5, 3))
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*max_len"):
        resolve("linear(3)\npos_embed(4)", input_shape=("B", 9, 2), output_shape=("B", 9, 3))
    # Exactly max_len positions are allowed.
    plan = resolve("pos_embed(5)", input_shape=("B", 5, 3), output_shape=("B", 5, 3))
    assert plan.nodes[0].args["max_len"] == 5


def test_the_width_flows_through_in_both_directions():
    backward = resolve("linear()\npos_embed(16)", input_shape=("B", 4, 6), output_shape=("B", 4, 9))
    assert backward.nodes[0].args["out_features"] == 9
    forward = resolve("pos_embed(16)\nlinear()", input_shape=("B", 4, 6), output_shape=("B", 4, 3))
    assert forward.nodes[0].output_shapes["out"] == ("B", 4, 6)
    assert forward.nodes[1].args["in_features"] == 6


@pytest.mark.parametrize("source", ["pos_embed(0)", "pos_embed()"])
def test_invalid_max_len_reports_e_argument(source):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(source, input_shape=("B", 4, 3), output_shape=("B", 4, 3))


def test_rank_two_and_rank_four_inputs_are_rejected():
    for shape in (("B", 12), ("B", 2, 3, 4)):
        with pytest.raises(HNDLError, match="E_CONSTRAINT"):
            resolve("pos_embed(8)", input_shape=shape, output_shape=shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_build_on_cuda(dtype):
    plan = resolve("pos_embed(32)\nlinear(4)", input_shape=("B", 6, 8), output_shape=("B", 6, 4), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=3)
    torch_dtype = getattr(torch, dtype)
    assert model["n0"].weight.dtype == torch_dtype
    x = torch.randn(2, 6, 8, device="cuda:0", dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (2, 6, 4)
    output.float().square().mean().backward()
    assert model["n0"].weight.grad.isfinite().all()
