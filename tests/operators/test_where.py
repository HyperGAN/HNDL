"""Numerics, inference and error codes for the ``where`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "where")


def test_where_ternary_selection(device):
    plan = resolve(
        "out = where(cond, a, b)",
        input_shape={"cond": ("B", 4, 8), "a": ("B", 4, 8), "b": ("B", 4, 8)},
        input_dtype={"cond": "bool", "a": "float32", "b": "float32"},
        output_shape={"out": ("B", 4, 8)},
    )
    model = build(plan, device=device)
    cond = torch.tensor([[[True, False] * 4] * 4, [[False, True] * 4] * 4], device=device)
    a = torch.ones(2, 4, 8, device=device) * 10.0
    b = torch.ones(2, 4, 8, device=device) * -5.0
    out = model(cond=cond, a=a, b=b)["out"]
    assert out.shape == (2, 4, 8)
    expected = torch.where(cond, a, b)
    torch.testing.assert_close(out, expected)


def test_where_gradient_flow(device):
    plan = resolve(
        "out = where(cond, a, b)",
        input_shape={"cond": ("B", 4), "a": ("B", 4), "b": ("B", 4)},
        input_dtype={"cond": "bool", "a": "float32", "b": "float32"},
        output_shape={"out": ("B", 4)},
    )
    model = build(plan, device=device)
    cond = torch.tensor([[True, False, True, False]], device=device)
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device, requires_grad=True)
    b = torch.tensor([[5.0, 6.0, 7.0, 8.0]], device=device, requires_grad=True)
    out = model(cond=cond, a=a, b=b)["out"]
    out.sum().backward()
    assert a.grad is not None and b.grad is not None
    torch.testing.assert_close(a.grad, torch.tensor([[1.0, 0.0, 1.0, 0.0]], device=device))
    torch.testing.assert_close(b.grad, torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=device))


def test_where_rejects_non_bool_cond():
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve(
            "out = where(cond, a, b)",
            input_shape={"cond": ("B", 4), "a": ("B", 4), "b": ("B", 4)},
            input_dtype={"cond": "float32", "a": "float32", "b": "float32"},
            output_shape={"out": ("B", 4)},
        )


def test_where_rank_5(device):
    plan = resolve(
        "out = where(cond, a, b)",
        input_shape={"cond": ("B", 2, 4, 8, 16), "a": ("B", 2, 4, 8, 16), "b": ("B", 2, 4, 8, 16)},
        input_dtype={"cond": "bool", "a": "float32", "b": "float32"},
        output_shape={"out": ("B", 2, 4, 8, 16)},
    )
    model = build(plan, device=device)
    cond = torch.rand(1, 2, 4, 8, 16, device=device) > 0.5
    a = torch.randn(1, 2, 4, 8, 16, device=device)
    b = torch.randn(1, 2, 4, 8, 16, device=device)
    out = model(cond=cond, a=a, b=b)["out"]
    assert out.shape == (1, 2, 4, 8, 16)
    torch.testing.assert_close(out, torch.where(cond, a, b))
