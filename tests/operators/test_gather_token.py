"""Numerics, inference, and error codes specific to ``gather_token``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build

SOURCE = "tokens = embedding(x, 32, 8)\nfeatures = linear(tokens, 8)\ngather_token(features, x)"


def gathered(x, ids):
    """A handwritten gather: row ``argmax(ids[b])`` of each example."""
    return torch.stack([x[b, int(ids[b].argmax())] for b in range(x.shape[0])])


def gather_module(tokens, width, device, seed=None):
    """The bare ``gather_token`` module out of a minimal embed-and-pool plan."""
    plan = resolve(f"tokens = embedding(x, 64, {width})\ngather_token(tokens, x)",
                   input_shape=("B", tokens), output_shape=("B", width), input_dtype="int64")
    return build(plan, device=device, initialization_seed=seed)["n1"]


@pytest.mark.parametrize("batch,tokens,width", [(1, 1, 3), (2, 7, 5), (4, 3, 16)])
def test_gather_matches_a_handwritten_gather_with_gradients(batch, tokens, width, device):
    module = gather_module(tokens, width, device)
    x = torch.randn(batch, tokens, width, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    ids = torch.randint(0, 50, (batch, tokens), device=device)

    actual, reference = module(x, ids), gathered(mirror, ids)
    assert actual.shape == (batch, width)
    torch.testing.assert_close(actual, reference)

    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_the_whole_network_pools_the_position_of_the_largest_id(device):
    plan = resolve(SOURCE, input_shape=("B", 6), output_shape=("B", 8), input_dtype="int64")
    model = build(plan, device=device, initialization_seed=7)
    ids = torch.tensor([[5, 9, 31, 0, 0, 0], [31, 1, 2, 3, 4, 5]], device=device)
    pooled = model(x=ids)["output"]
    sequence = model["n1"](model["n0"](ids))
    assert pooled.shape == (2, 8)
    torch.testing.assert_close(pooled, torch.stack([sequence[0, 2], sequence[1, 0]]))


def test_unselected_positions_receive_no_gradient(device):
    module = gather_module(4, 3, device)
    x = torch.randn(2, 4, 3, device=device, requires_grad=True)
    ids = torch.tensor([[0, 7, 0, 0], [0, 0, 0, 7]], device=device)
    module(x, ids).square().sum().backward()
    assert (x.grad[0, 1] != 0).any() and (x.grad[1, 3] != 0).any()
    assert torch.equal(x.grad[0, [0, 2, 3]], torch.zeros(3, 3, device=device))
    assert torch.equal(x.grad[1, :3], torch.zeros(3, 3, device=device))


def test_ties_select_the_first_maximum(device):
    module = gather_module(4, 2, device)
    x = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]] * 2, device=device)
    ids = torch.tensor([[3, 9, 9, 9], [5, 5, 5, 5]], device=device)
    torch.testing.assert_close(module(x, ids), torch.tensor([[2.0, 3.0], [0.0, 1.0]], device=device))


def test_default_mode_is_argmax_and_is_recorded_in_the_plan():
    plan = resolve("tokens = embedding(x, 32, 4)\ngather_token(tokens, x)",
                   input_shape=("B", 5), output_shape=("B", 4), input_dtype="int64")
    assert plan.nodes[1].args["mode"] == "argmax"
    assert "mode='argmax'" in repr(build(plan, device="cpu")["n1"])


@pytest.mark.parametrize("call,code", [
    ('gather_token(tokens, x, mode="argmax")', None),
    ('gather_token(tokens, x, "argmax")', "E_ARGUMENT"),
    ('gather_token(tokens, x, mode="last")', "E_ARGUMENT"),
    ("gather_token(tokens, x, mode=1)", "E_ARGUMENT"),
])
def test_mode_is_keyword_only_and_validated(call, code):
    source = f"tokens = embedding(x, 32, 4)\n{call}"
    contract = dict(input_shape=("B", 5), output_shape=("B", 4), input_dtype="int64")
    if code is None:
        assert resolve(source, **contract).nodes[1].args["mode"] == "argmax"
        return
    with pytest.raises(HNDLError, match=code):
        resolve(source, **contract)


def test_float_ids_are_rejected_at_resolution():
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("features = linear(x, 4)\ngather_token(features, x)",
                input_shape=("B", 6, 3), output_shape=("B", 4))
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("tokens = embedding(x, 32, 8)\ngather_token(tokens, tokens)",
                input_shape=("B", 5), output_shape=("B", 8), input_dtype="int64")


def test_a_token_count_mismatch_between_x_and_ids_is_a_contradiction():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("tokens = embedding(x, 32, 8)\nshort, rest = split(tokens, 3)\ngather_token(short, x)",
                input_shape=("B", 6), output_shape=("B", 8), input_dtype="int64")


def test_width_flows_backward_from_the_pooled_output():
    plan = resolve("tokens = embedding(x, 32, 8)\nfeatures = linear(tokens)\ngather_token(features, x)",
                   input_shape=("B", 6), output_shape=("B", 5), input_dtype="int64")
    assert plan.nodes[1].args["out_features"] == 5
    assert plan.nodes[2].input_shapes["x"] == ("B", 6, 5)
    assert plan.nodes[2].output_shapes["out"] == ("B", 5)


def test_width_flows_forward_into_the_consumer():
    plan = resolve("tokens = embedding(x, 32, 9)\npooled = gather_token(tokens, x)\nlinear(pooled)",
                   input_shape=("B", 6), output_shape=("B", 3), input_dtype="int64")
    assert plan.nodes[1].output_shapes["out"] == ("B", 9)
    assert plan.nodes[2].args["in_features"] == 9


@pytest.mark.parametrize("output_shape", [("B", 4, 8), ("B", 8, 8, 8)])
def test_a_non_vector_output_is_a_shape_contradiction(output_shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("tokens = embedding(x, 32, 8)\ngather_token(tokens, x)",
                input_shape=("B", 6), output_shape=output_shape, input_dtype="int64")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_runs_in_reduced_precision_on_cuda(dtype):
    plan = resolve(SOURCE, input_shape=("B", 6), output_shape=("B", 8),
                   input_dtype="int64", dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=3)
    torch_dtype = getattr(torch, dtype)
    ids = torch.randint(0, 32, (2, 6), device="cuda:0")
    pooled = model(x=ids)["output"]
    assert pooled.dtype == torch_dtype and pooled.shape == (2, 8)

    sequence = model["n1"](model["n0"](ids))
    torch.testing.assert_close(pooled, gathered(sequence, ids))
    pooled.float().square().mean().backward()
    assert model["n0"].weight.grad is not None and model["n0"].weight.grad.isfinite().all()
