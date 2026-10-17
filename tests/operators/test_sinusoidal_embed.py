"""Numerics, inference and error codes for the ``sinusoidal_embed`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "sinusoidal_embed")


def test_matches_sinusoidal_formula_for_scalar_input(device):
    plan = resolve("sinusoidal_embed(32)", input_shape=("B", 1), output_shape=("B", 32))
    model = build(plan, device=device)
    x = torch.tensor([[0.0], [1.0], [50.0]], device=device)
    out = model(x=x)["output"]
    # At t=0, sin(0) is 0 and cos(0) is 1
    torch.testing.assert_close(out[0, :16], torch.zeros(16, device=device))
    torch.testing.assert_close(out[0, 16:], torch.ones(16, device=device))


def test_supports_multidimensional_coordinate_inputs(device):
    plan = resolve("sinusoidal_embed(64)", input_shape=("B", 2), output_shape=("B", 64))
    model = build(plan, device=device)
    x = torch.randn(4, 2, device=device)
    out = model(x=x)["output"]
    assert out.shape == (4, 64)
    assert out.isfinite().all()


def test_features_is_inferred_forward_and_backward():
    forward = resolve("sinusoidal_embed(64)\nlinear(10)", input_shape=("B", 1), output_shape=("B", 10))
    assert forward.nodes[0].output_shapes["out"] == ("B", 64)
    assert forward.nodes[1].args["in_features"] == 64

    backward = resolve("sinusoidal_embed()", input_shape=("B", 1), output_shape=("B", 48))
    assert backward.nodes[0].args["features"] == 48


def test_odd_features_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*even"):
        resolve("sinusoidal_embed(33)", input_shape=("B", 1), output_shape=("B", 33))


def test_features_not_divisible_by_two_times_d_in_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*multiple"):
        resolve("sinusoidal_embed(10)", input_shape=("B", 3), output_shape=("B", 10))


def test_rejects_rank_3_and_rank_4_inputs():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("sinusoidal_embed(32)", input_shape=("B", 10, 1), output_shape=("B", 10, 32))


def test_gradcheck_double(device):
    plan = resolve("sinusoidal_embed(16)", input_shape=("B", 1), output_shape=("B", 16))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    x = torch.randn(2, 1, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve("sinusoidal_embed(32)", input_shape=("B", 1), output_shape=("B", 32), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 1, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
