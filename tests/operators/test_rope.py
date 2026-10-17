"""Numerics, inference and error codes for the ``rope`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "rope")


def test_position_zero_is_unrotated(device):
    plan = resolve("rope(heads=2)", input_shape=("B", 8, 16), output_shape=("B", 8, 16))
    model = build(plan, device=device)
    x = torch.randn(2, 8, 16, device=device)
    out = model(x=x)["output"]
    # At t=0, angles are 0 so sin(0)=0 and cos(0)=1
    torch.testing.assert_close(out[:, 0], x[:, 0])


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("theta", [10000.0, 500000.0])
def test_matches_handwritten_rotary_embeddings(device, heads, theta):
    plan = resolve(f"rope(heads={heads}, theta={theta})", input_shape=("B", 6, 24), output_shape=("B", 6, 24))
    model = build(plan, device=device)
    x = torch.randn(3, 6, 24, device=device)
    actual = model(x=x)["output"]

    # Manual computation
    b, t, d = x.shape
    dh = d // heads
    calc_dtype = torch.float32
    exponent = torch.arange(0, dh, 2, device=device, dtype=calc_dtype) / dh
    inv_freq = theta ** (-exponent)
    angles = torch.arange(t, device=device, dtype=calc_dtype)[:, None] * inv_freq[None, :]
    dup = torch.cat((angles, angles), dim=-1)
    cos = dup.cos().to(x.dtype)[None, :, None, :]
    sin = dup.sin().to(x.dtype)[None, :, None, :]
    xh = x.view(b, t, heads, dh)
    half = dh // 2
    rot = torch.cat((-xh[..., half:], xh[..., :half]), dim=-1)
    expected = (xh * cos + rot * sin).view(b, t, d)
    torch.testing.assert_close(actual, expected)


def test_length_and_width_flow_bidirectionally():
    forward = resolve("rope(2)\nlinear(10)", input_shape=("B", 12, 16), output_shape=("B", 12, 10))
    assert forward.nodes[0].output_shapes["out"] == ("B", 12, 16)
    assert forward.nodes[1].args["in_features"] == 16

    backward = resolve("linear()\nrope(4)", input_shape=("B", 10, 8), output_shape=("B", 10, 32))
    assert backward.nodes[0].args["out_features"] == 32


def test_odd_head_dimension_reports_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*even"):
        resolve("rope(heads=3)", input_shape=("B", 8, 9), output_shape=("B", 8, 9))


def test_heads_not_dividing_d_reports_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divide"):
        resolve("rope(heads=5)", input_shape=("B", 8, 16), output_shape=("B", 8, 16))


def test_rejects_rank_2_and_rank_4():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("rope()", input_shape=("B", 16), output_shape=("B", 16))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("rope()", input_shape=("B", 4, 8, 8), output_shape=("B", 4, 8, 8))


def test_gradcheck_double(device):
    plan = resolve("rope(heads=2)", input_shape=("B", 4, 8), output_shape=("B", 4, 8))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    x = torch.randn(2, 4, 8, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve("rope(heads=2)", input_shape=("B", 8, 16), output_shape=("B", 8, 16), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 8, 16, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
