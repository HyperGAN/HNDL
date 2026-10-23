"""Numerics, inference and error codes for the ``drop_path`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "drop_path")


def test_drop_path_eval_mode_is_exact_identity(device):
    plan = resolve(
        "drop_path(0.5)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    model.eval()
    x = torch.randn(4, 8, 16, device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, x)


def test_drop_path_zero_prob_is_identity_in_train(device):
    plan = resolve(
        "drop_path(0.0)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    model.train()
    x = torch.randn(4, 8, 16, device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, x)


def test_drop_path_one_prob_drops_all_in_train(device):
    plan = resolve(
        "drop_path(1.0)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    model.train()
    x = torch.randn(4, 8, 16, device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, torch.zeros_like(x))


def test_drop_path_intermediate_scaling(device):
    plan = resolve(
        "drop_path(0.5, scale_by_keep=True)",
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model = build(plan, device=device)
    model.train()
    torch.manual_seed(42)
    x = torch.ones(1000, 4, 8, device=device)
    out = model(x=x)["output"]
    # For kept samples, value is 1.0 / 0.5 = 2.0. For dropped samples, 0.0.
    # Mean of output across large batch should be close to 1.0.
    mean_val = out.mean().item()
    assert abs(mean_val - 1.0) < 0.1
    # Check that each sample across batch is either all 0 or all 2.0
    sample_slices = out[:, 0, 0]
    for s in sample_slices[:20]:
        assert s.item() == 0.0 or abs(s.item() - 2.0) < 1e-5


def test_drop_path_scale_by_keep_false(device):
    plan = resolve(
        "drop_path(0.5, scale_by_keep=False)",
        input_shape=("B", 4, 8),
        output_shape=("B", 4, 8),
    )
    model = build(plan, device=device)
    model.train()
    torch.manual_seed(42)
    x = torch.ones(100, 4, 8, device=device)
    out = model(x=x)["output"]
    sample_slices = out[:, 0, 0]
    for s in sample_slices[:20]:
        assert s.item() == 0.0 or abs(s.item() - 1.0) < 1e-5


def test_drop_path_gradient_flow(device):
    plan = resolve(
        "drop_path(0.0)",
        input_shape=("B", 8, 16),
        output_shape=("B", 8, 16),
    )
    model = build(plan, device=device)
    model.train()
    x = torch.randn(2, 8, 16, device=device, requires_grad=True)
    out = model(x=x)["output"]
    out.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_drop_path_rejects_invalid_probability():
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(
            "drop_path(-0.1)",
            input_shape=("B", 8, 16),
            output_shape=("B", 8, 16),
        )
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(
            "drop_path(1.5)",
            input_shape=("B", 8, 16),
            output_shape=("B", 8, 16),
        )
