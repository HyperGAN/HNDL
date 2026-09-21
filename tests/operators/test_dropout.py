"""Numerics, train/eval behavior, and error codes for ``dropout``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build


def _module(plan, model):
    node = next(node for node in plan.nodes if node.op.startswith("dropout@"))
    return model[node.id]


def test_p_zero_is_exactly_the_identity_in_both_modes(device):
    plan = resolve("dropout(0.0)", input_shape=("B", 4, 6), output_shape=("B", 4, 6))
    model = build(plan, device=device)
    x = torch.randn(5, 4, 6, device=device)
    torch.testing.assert_close(model(x=x)["output"], x)
    model.eval()
    torch.testing.assert_close(model(x=x)["output"], x)


def test_eval_mode_is_the_identity_and_deterministic_for_every_p(device):
    plan = resolve("dropout(0.75)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 8, 8))
    model = build(plan, device=device)
    model.eval()
    x = torch.randn(4, 3, 8, 8, device=device)
    first = model(x=x)["output"]
    second = model(x=x)["output"]
    torch.testing.assert_close(first, x)
    torch.testing.assert_close(first, second)


def test_training_mode_zeros_about_p_and_rescales_survivors(device):
    p = 0.4
    plan = resolve(f"dropout({p})", input_shape=("B", 512), output_shape=("B", 512))
    model = build(plan, device=device)
    assert model.training
    x = torch.full((256, 512), 3.0, device=device)
    out = model(x=x)["output"]
    dropped = out == 0
    assert 0.35 < dropped.float().mean().item() < 0.45
    survivors = out[~dropped]
    torch.testing.assert_close(survivors, torch.full_like(survivors, 3.0 / (1.0 - p)))
    # The expectation of the masked output is the input.
    assert abs(out.mean().item() - 3.0) < 0.05


def test_training_mode_draws_a_fresh_mask_per_call_but_reseeding_reproduces_it(device):
    plan = resolve("dropout(0.5)", input_shape=("B", 64), output_shape=("B", 64))
    model = build(plan, device=device)
    x = torch.randn(32, 64, device=device)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    first = model(x=x)["output"]
    second = model(x=x)["output"]
    assert not torch.equal(first, second)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.testing.assert_close(model(x=x)["output"], first)


def test_gradients_follow_the_mask(device):
    plan = resolve("dropout(0.5)", input_shape=("B", 128), output_shape=("B", 128))
    model = build(plan, device=device)
    x = torch.randn(16, 128, device=device, requires_grad=True)
    out = model(x=x)["output"]
    out.sum().backward()
    torch.testing.assert_close(x.grad, (out != 0).to(x.dtype) * 2.0)


@pytest.mark.parametrize("shape", [("B", 7), ("B", 5, 9), ("B", 2, 6, 6)])
def test_shape_is_preserved_on_every_supported_rank(shape, device):
    plan = resolve("dropout(0.25)", input_shape=shape, output_shape=shape)
    assert plan.nodes[0].output_shapes["out"] == shape
    model = build(plan, device=device)
    x = torch.randn(3, *shape[1:], device=device)
    assert tuple(model(x=x)["output"].shape) == (3, *shape[1:])


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_mask_is_applied_in_the_compute_dtype(dtype):
    if dtype != "float32" and not torch.cuda.is_available():
        pytest.skip("reduced precision kernels are qualified on CUDA")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    plan = resolve("dropout(0.3)", input_shape=("B", 256), output_shape=("B", 256), dtype=dtype)
    model = build(plan, device=device)
    x = torch.randn(8, 256, device=device, dtype=DTYPES[dtype])
    out = model(x=x)["output"]
    assert out.dtype == DTYPES[dtype]
    assert (out == 0).any() and out.isfinite().all()


def test_matches_the_functional_reference_under_a_shared_seed(device):
    plan = resolve("dropout(0.6)", input_shape=("B", 10, 10), output_shape=("B", 10, 10))
    model = build(plan, device=device)
    module = _module(plan, model)
    x = torch.randn(4, 10, 10, device=device)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    actual = model(x=x)["output"]
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    expected = torch.nn.functional.dropout(x, module.p, True)
    torch.testing.assert_close(actual, expected)


def test_the_module_holds_no_parameters_or_buffers(device):
    plan = resolve("dropout(0.1)", input_shape=("B", 4), output_shape=("B", 4))
    model = build(plan, device=device)
    module = _module(plan, model)
    assert list(module.parameters()) == [] and list(module.buffers()) == []


def test_shapes_flow_backward_through_dropout():
    plan = resolve("linear(64)\ndropout(0.2)\nlinear()", input_shape=("B", 16), output_shape=("B", 3))
    assert [node.output_shapes["out"] for node in plan.nodes] == [("B", 64), ("B", 64), ("B", 3)]
    assert plan.nodes[2].args["in_features"] == 64
    forward = resolve("dropout(0.2)\nlinear()", input_shape=("B", 5, 12), output_shape=("B", 5, 4))
    assert forward.nodes[0].input_shapes["x"] == ("B", 5, 12)
    assert forward.nodes[1].args["in_features"] == 12
    backward = resolve("linear()\ndropout(0.2)", input_shape=("B", 12), output_shape=("B", 7))
    assert backward.nodes[0].args["out_features"] == 7


def test_the_probability_defaults_to_a_tenth_and_round_trips():
    plan = resolve("dropout()", input_shape=("B", 4), output_shape=("B", 4))
    assert plan.nodes[0].args["p"] == 0.1
    named = resolve("dropout(p=0.35)", input_shape=("B", 4), output_shape=("B", 4))
    assert named.nodes[0].args["p"] == 0.35
    assert named.semantic_digest != plan.semantic_digest


@pytest.mark.parametrize("source", ["dropout(1.0)", "dropout(1)", "dropout(2.5)"])
def test_p_at_or_above_one_is_rejected(source):
    with pytest.raises(HNDLError, match="E_ARGUMENT") as info:
        resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    assert info.value.code == "E_ARGUMENT"


@pytest.mark.parametrize("source", ["dropout(-0.1)", "dropout(p=-1.0)"])
def test_negative_p_is_rejected(source):
    with pytest.raises(HNDLError) as info:
        resolve(source, input_shape=("B", 4), output_shape=("B", 4))
    assert info.value.code == "E_ARGUMENT"


def test_a_non_numeric_probability_is_rejected():
    with pytest.raises(HNDLError) as info:
        resolve("dropout('half')", input_shape=("B", 4), output_shape=("B", 4))
    assert info.value.code == "E_ARGUMENT"


def test_dropout_cannot_change_the_shape():
    with pytest.raises(HNDLError) as info:
        resolve("dropout(0.1)", input_shape=("B", 4), output_shape=("B", 5))
    assert info.value.code == "E_CONSTRAINT"
