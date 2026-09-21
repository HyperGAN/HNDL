"""Numerics, inference and error codes for the ``feed_forward`` block."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build, network, parameter_counts

ACTIVATIONS = {
    "gelu": lambda h: F.gelu(h),
    "gelu_tanh": lambda h: F.gelu(h, approximate="tanh"),
    "relu": lambda h: torch.where(h > 0, h, torch.zeros_like(h)),
    "silu": lambda h: h * torch.sigmoid(h),
    "quick_gelu": lambda h: h * torch.sigmoid(1.702 * h),
}


def handwritten(module, x, activation):
    hidden = ACTIVATIONS[activation](F.linear(x, module.up.weight, module.up.bias))
    return F.linear(hidden, module.down.weight, module.down.bias)


@pytest.mark.parametrize("shape", [("B", 16), ("B", 5, 16)])
def test_matches_a_handwritten_block_on_rank_2_and_rank_3(device, shape):
    model = network("feed_forward(24)", input_shape=shape, output_shape=shape,
                    device=device, initialization_seed=7)
    module = model["n0"]
    x = torch.randn(3, *shape[1:], device=device)
    torch.testing.assert_close(model(x), handwritten(module, x, "gelu"))


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_every_activation_choice_matches_its_formula(device, activation):
    shape = ("B", 12)
    model = network(f'feed_forward(20, activation="{activation}")', input_shape=shape, output_shape=shape,
                    device=device, initialization_seed=3)
    module = model["n0"]
    assert module.activation == activation
    x = torch.randn(4, 12, device=device)
    torch.testing.assert_close(model(x), handwritten(module, x, activation))


def test_gradients_reach_both_projections(device):
    model = network("feed_forward(16)", input_shape=("B", 8), output_shape=("B", 8),
                    device=device, initialization_seed=1)
    x = torch.randn(2, 8, device=device, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    for name in ("up", "down"):
        weight = getattr(model["n0"], name).weight
        assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_dropout_is_stochastic_in_train_mode_and_identity_in_eval(device):
    model = network("feed_forward(64, dropout=0.5)", input_shape=("B", 16), output_shape=("B", 16),
                    device=device, initialization_seed=2)
    module = model["n0"]
    assert module.dropout.p == 0.5
    x = torch.randn(4, 16, device=device)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    torch.testing.assert_close(model(x), model(x))
    torch.testing.assert_close(model(x), handwritten(module, x, "gelu"))


def test_widths_are_inferred_forward_and_backward():
    plan = resolve("linear()\nfeed_forward(16)", input_shape=("B", 12), output_shape=("B", 7))
    assert plan.nodes[0].args["out_features"] == 7
    assert plan.nodes[1].input_shapes["x"] == ("B", 7) and plan.nodes[1].output_shapes["out"] == ("B", 7)
    forward = resolve("feed_forward(16)\nlinear()", input_shape=("B", 9), output_shape=("B", 3))
    assert forward.nodes[0].output_shapes["out"] == ("B", 9)
    sequence = resolve("feed_forward(16)", input_shape=("B", 5, 9), output_shape=("B", 5, 9))
    assert sequence.nodes[0].output_shapes["out"] == ("B", 5, 9)


@pytest.mark.parametrize("bias,expected", [(True, 2 * 16 * 24 + 24 + 16), (False, 2 * 16 * 24)])
def test_parameter_counts(bias, expected):
    plan = resolve(f"feed_forward(24, bias={bias})", input_shape=("B", 16), output_shape=("B", 16))
    assert parameter_counts(plan) == {"n0": expected}
    model = build(plan, device="cpu")
    assert sum(p.numel() for p in model.parameters()) == expected
    assert (model["n0"].up.bias is not None) is bias


@pytest.mark.parametrize("source,code,message", [
    ('feed_forward(8, activation="mish")', "E_ARGUMENT", "activation"),
    ("feed_forward(0)", "E_ARGUMENT", "hidden"),
    ("feed_forward(8, dropout=1.0)", "E_ARGUMENT", "dropout"),
    ("feed_forward(8, dropout=-0.5)", "E_ARGUMENT", "dropout"),
])
def test_invalid_arguments_report_e_argument(source, code, message):
    with pytest.raises(HNDLError, match=f"{code}.*{message}"):
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))


def test_images_are_rejected_and_width_changes_contradict():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*flatten"):
        resolve("feed_forward(8)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 4, 4))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("feed_forward(8)", input_shape=("B", 16), output_shape=("B", 10))


def test_hidden_is_required():
    with pytest.raises(HNDLError):
        resolve("feed_forward()", input_shape=("B", 8), output_shape=("B", 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve('feed_forward(32, activation="quick_gelu")', input_shape=("B", 5, 16),
                   output_shape=("B", 5, 16), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=4)
    torch_dtype = DTYPES[dtype]
    assert all(p.dtype == torch_dtype for p in model.parameters())
    x = torch.randn(2, 5, 16, device="cuda:0", dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (2, 5, 16)
    assert output.isfinite().all()
    expected = handwritten(model["n0"], x, "quick_gelu")
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
