"""Numerics, inference and error codes for the ``snake`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "snake")


def test_snake_periodic_formula(device):
    plan = resolve(
        "snake(alpha=2.0)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    x = torch.linspace(-3.0, 3.0, 16, device=device).unsqueeze(0).unsqueeze(0).repeat(2, 8, 1)
    out = model(x=x)["output"]

    alpha = 2.0
    eps = 1e-9
    expected = x + (1.0 / (alpha + eps)) * (torch.sin(alpha * x) ** 2)
    torch.testing.assert_close(out, expected)


def test_snake_channels_first_and_last(device):
    # channels-first [B, C, T]
    plan_first = resolve(
        "snake(channels=16)",
        input_shape=("B", 16, 32),
        output_shape=("B", 16, 32),
    )
    model_first = build(plan_first, device=device)
    x_first = torch.randn(2, 16, 32, device=device)
    out_first = model_first(x=x_first)["output"]
    assert out_first.shape == (2, 16, 32)
    assert out_first.isfinite().all()

    # channels-last [B, T, C]
    plan_last = resolve(
        "snake(channels=16)",
        input_shape=("B", 32, 16),
        output_shape=("B", 32, 16),
    )
    model_last = build(plan_last, device=device)
    x_last = torch.randn(2, 32, 16, device=device)
    out_last = model_last(x=x_last)["output"]
    assert out_last.shape == (2, 32, 16)
    assert out_last.isfinite().all()


def test_snake_dim_alias(device):
    plan = resolve(
        "snake(dim=16)",
        input_shape=("B", 32, 16),
        output_shape=("B", 32, 16),
    )
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    assert module.channels == 16
    assert module.alpha.shape == (16,)


def test_snake_gradients(device):
    plan = resolve(
        "snake(channels=8, alpha=1.5)",
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 4, 8, device=device, requires_grad=True)
    out = model(x=x)["output"]
    out.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    module = model[plan.nodes[0].id]
    assert module.alpha.grad is not None and module.alpha.grad.isfinite().all()


def test_snake_numerical_stability_near_zero(device):
    # Tests that when alpha is very small or eps prevents division by zero
    plan = resolve(
        "snake(alpha=0.001, eps=1e-6)",
        input_shape=("B", 8),
        output_shape=("B", 8),
    )
    model = build(plan, device=device)
    x = torch.randn(2, 8, device=device)
    out = model(x=x)["output"]
    assert out.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve(
        "snake(channels=16)",
        input_shape=("B", 16, 32),
        output_shape=("B", 16, 32),
        dtype=dtype,
    )
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 16, 32, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()


def test_snake_mismatched_channels_raises_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(
            "snake(channels=16)",
            input_shape=("B", 32, 64),
            output_shape=("B", 32, 64),
        )
