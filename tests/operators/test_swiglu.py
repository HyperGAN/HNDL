"""Numerics, inference and error codes for the ``swiglu`` block."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build, network, parameter_counts


def handwritten(module, x):
    gate = F.linear(x, module.gate.weight, module.gate.bias)
    up = F.linear(x, module.up.weight, module.up.bias)
    return F.linear((gate * torch.sigmoid(gate)) * up, module.down.weight, module.down.bias)


@pytest.mark.parametrize("shape", [("B", 16), ("B", 5, 16)])
def test_matches_a_handwritten_block_on_rank_2_and_rank_3(device, shape):
    model = network("swiglu(24)", input_shape=shape, output_shape=shape, device=device, initialization_seed=7)
    x = torch.randn(3, *shape[1:], device=device)
    torch.testing.assert_close(model(x), handwritten(model["n0"], x))


def test_biased_block_matches_the_handwritten_form(device):
    model = network("swiglu(20, bias=True)", input_shape=("B", 12), output_shape=("B", 12),
                    device=device, initialization_seed=5)
    module = model["n0"]
    assert all(getattr(module, name).bias is not None for name in ("gate", "up", "down"))
    x = torch.randn(4, 12, device=device)
    torch.testing.assert_close(model(x), handwritten(module, x))


def test_the_gate_is_silu_and_only_the_gate_branch_is_activated(device):
    model = network("swiglu(8)", input_shape=("B", 6), output_shape=("B", 6),
                    device=device, initialization_seed=11)
    module = model["n0"]
    x = torch.randn(3, 6, device=device)
    with torch.no_grad():
        module.gate.weight.zero_()
    # silu(0) == 0, so a zeroed gate annihilates the block whatever `up` holds.
    torch.testing.assert_close(model(x), torch.zeros(3, 6, device=device))


def test_gradients_reach_all_three_projections(device):
    model = network("swiglu(16)", input_shape=("B", 8), output_shape=("B", 8),
                    device=device, initialization_seed=1)
    x = torch.randn(2, 8, device=device, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    for name in ("gate", "up", "down"):
        weight = getattr(model["n0"], name).weight
        assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_train_and_eval_agree(device):
    model = network("swiglu(16)", input_shape=("B", 8), output_shape=("B", 8),
                    device=device, initialization_seed=1)
    x = torch.randn(2, 8, device=device)
    model.train()
    trained = model(x)
    model.eval()
    torch.testing.assert_close(trained, model(x))


def test_widths_are_inferred_forward_and_backward():
    plan = resolve("linear()\nswiglu(16)", input_shape=("B", 12), output_shape=("B", 7))
    assert plan.nodes[0].args["out_features"] == 7
    assert plan.nodes[1].input_shapes["x"] == ("B", 7) and plan.nodes[1].output_shapes["out"] == ("B", 7)
    forward = resolve("swiglu(16)\nlinear()", input_shape=("B", 9), output_shape=("B", 3))
    assert forward.nodes[0].output_shapes["out"] == ("B", 9)
    sequence = resolve("swiglu(16)", input_shape=("B", 5, 9), output_shape=("B", 5, 9))
    assert sequence.nodes[0].output_shapes["out"] == ("B", 5, 9)


@pytest.mark.parametrize("bias,expected", [(False, 3 * 16 * 24), (True, 3 * 16 * 24 + 2 * 24 + 16)])
def test_parameter_counts(bias, expected):
    plan = resolve(f"swiglu(24, bias={bias})", input_shape=("B", 16), output_shape=("B", 16))
    assert parameter_counts(plan) == {"n0": expected}
    model = build(plan, device="cpu")
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize("source,message", [
    ("swiglu(0)", "hidden"),
    ('swiglu(8, bias="yes")', "bias"),
])
def test_invalid_arguments_report_e_argument(source, message):
    with pytest.raises(HNDLError, match=f"E_ARGUMENT.*{message}"):
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))


def test_images_are_rejected_and_width_changes_contradict():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*flatten"):
        resolve("swiglu(8)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 4, 4))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("swiglu(8)", input_shape=("B", 16), output_shape=("B", 10))


def test_hidden_is_required():
    with pytest.raises(HNDLError):
        resolve("swiglu()", input_shape=("B", 8), output_shape=("B", 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve("swiglu(32)", input_shape=("B", 5, 16), output_shape=("B", 5, 16), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=4)
    torch_dtype = DTYPES[dtype]
    assert all(p.dtype == torch_dtype for p in model.parameters())
    x = torch.randn(2, 5, 16, device="cuda:0", dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (2, 5, 16)
    assert output.isfinite().all()
    torch.testing.assert_close(output, handwritten(model["n0"], x), rtol=2e-2, atol=2e-2)
