"""Numerics, inference, and error codes specific to ``embedding``."""

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.torch import build, network


def test_embedding_matches_nn_embedding_forward_and_backward(device):
    model = network("embedding(50, 12)", input_shape=("B", 7), output_shape=("B", 7, 12),
                    input_dtype="int64", device=device, initialization_seed=11)
    table = model["n0"]
    assert isinstance(table, nn.Embedding)
    assert tuple(table.weight.shape) == (50, 12)

    reference = nn.Embedding(50, 12).to(device)
    with torch.no_grad():
        reference.weight.copy_(table.weight)
    ids = torch.randint(0, 50, (4, 7), device=device)
    actual, expected = model(ids), reference(ids)
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(table.weight.grad, reference.weight.grad)


def test_unused_rows_receive_no_gradient(device):
    model = network("embedding(6, 4)", input_shape=("B", 3), output_shape=("B", 3, 4),
                    input_dtype="int64", device=device, initialization_seed=2)
    ids = torch.zeros(2, 3, dtype=torch.int64, device=device)
    model(ids).square().sum().backward()
    grad = model["n0"].weight.grad
    assert (grad[0] != 0).any()
    assert torch.equal(grad[1:], torch.zeros_like(grad[1:]))


def test_width_is_inferred_from_the_consumer_in_both_directions():
    plan = resolve("embedding(32)\nrelu()", input_shape=("B", 5), output_shape=("B", 5, 7),
                   input_dtype="int64")
    assert plan.nodes[0].args["dim"] == 7
    assert plan.nodes[0].output_shapes["out"] == ("B", 5, 7)

    forward = resolve("embedding(32, 9)\nlinear(4)", input_shape=("B", 5), output_shape=("B", 5, 4),
                      input_dtype="int64")
    assert forward.nodes[1].args["in_features"] == 9

    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("embedding(32)\nlinear(4)", input_shape=("B", 5), output_shape=("B", 5, 4),
                input_dtype="int64")


def test_float_graph_input_is_rejected_at_resolution():
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("embedding(32, 8)", input_shape=("B", 5), output_shape=("B", 5, 8))
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("relu()\nembedding(32, 8)", input_shape=("B", 5), output_shape=("B", 5, 8),
                input_dtype="int64")


def test_float_tensor_is_rejected_at_runtime(device):
    model = network("embedding(16, 4)", input_shape=("B", 3), output_shape=("B", 3, 4),
                    input_dtype="int64", device=device)
    ids = torch.randint(0, 16, (2, 3), device=device)
    assert model(ids).shape == (2, 3, 4)
    with pytest.raises(HNDLError, match="E_RUNTIME.*dtype"):
        model(ids.float())
    with pytest.raises(HNDLError, match="E_RUNTIME.*dtype"):
        model(ids.to(torch.int32))


@pytest.mark.parametrize("source,code", [
    ("embedding(0, 4)", "E_ARGUMENT"),
    ("embedding(16, 0)", "E_ARGUMENT"),
    ("embedding()", "E_ARGUMENT"),
])
def test_invalid_arguments_report_their_code(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 3), output_shape=("B", 3, 4), input_dtype="int64")


def test_rank_three_ids_are_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("embedding(16, 4)", input_shape=("B", 3, 2), output_shape=("B", 3, 2, 4),
                input_dtype="int64")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_build_on_cuda(dtype):
    plan = resolve("embedding(64, 8)\nlinear(4)", input_shape=("B", 5), output_shape=("B", 5, 4),
                   input_dtype="int64", dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=3)
    torch_dtype = getattr(torch, dtype)
    assert model["n0"].weight.dtype == torch_dtype
    ids = torch.randint(0, 64, (2, 5), device="cuda:0")
    output = model(x=ids)["output"]
    assert output.dtype == torch_dtype and output.shape == (2, 5, 4)
    output.float().square().mean().backward()
    assert model["n0"].weight.grad is not None and model["n0"].weight.grad.isfinite().all()
