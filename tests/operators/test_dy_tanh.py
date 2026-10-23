"""Numerics, inference and error codes for the ``dy_tanh`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "dy_tanh")


def test_dy_tanh_forward_and_backward(device):
    plan = resolve(
        "dy_tanh(dim=16, init_alpha=1.5, affine=True)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 8, 16, device=device, requires_grad=True)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 16)
    assert out.isfinite().all()

    out.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    # Check parameters have grads
    module = model[plan.nodes[0].id]
    assert module.alpha.grad is not None
    assert module.gamma.grad is not None
    assert module.beta.grad is not None


def test_dy_tanh_non_affine(device):
    plan = resolve(
        "dy_tanh(dim=16, affine=False)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 8, 16, device=device)
    out = model(x=x)["output"]
    expected = torch.tanh(x)
    torch.testing.assert_close(out, expected)


def test_dy_tanh_numerical_extremes(device):
    plan = resolve(
        "dy_tanh(dim=8, init_alpha=1.0, affine=False)",
        input_shape=("B", 8),
        output_shape=("B", 8),
    )
    model = build(plan, device=device)
    x = torch.tensor([[-1000.0, -100.0, 0.0, 100.0, 1000.0, 1e-6, -1e-6, 5.0]], device=device)
    out = model(x=x)["output"]
    assert out.isfinite().all()
    assert torch.allclose(out[0, 0], torch.tensor(-1.0, device=device))
    assert torch.allclose(out[0, 4], torch.tensor(1.0, device=device))
    assert torch.allclose(out[0, 2], torch.tensor(0.0, device=device))


def test_dy_tanh_shape_inference():
    plan = resolve(
        "linear(32)\ndy_tanh()",
        input_shape=("B", 16),
        output_shape=("B", 32),
    )
    assert plan.nodes[1].args["dim"] == 32
    assert plan.nodes[1].output_shapes["out"] == ("B", 32)


def test_dy_tanh_rejects_dim_mismatch():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(
            "dy_tanh(dim=16)",
            input_shape=("B", 8, 32),
            output_shape=("B", 8, 32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        "dy_tanh(dim=16)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 8, 16, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
