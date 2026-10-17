"""Numerics, dtype propagation, and error cases for the ``cast`` operator."""

import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "cast")


@pytest.mark.parametrize("target", ["float32", "float16", "bfloat16", "float64", "int64", "int32", "bool"])
def test_casts_to_all_supported_target_types(device, target):
    plan = resolve(f"cast('{target}')", input_shape=("B", 8), output_shape=("B", 8))
    model = build(plan, device=device)
    cast_module = model[plan.nodes[0].id]
    x = torch.randn(2, 8, device=device)
    out = cast_module(x)
    expected_dtype = getattr(torch, target)
    assert out.dtype == expected_dtype
    assert out.shape == (2, 8)


def test_compute_dtype_tracks_plan_dtype(device):
    for dtype in ("float32", "float16", "bfloat16"):
        if dtype in ("float16", "bfloat16") and device == "cpu":
            continue
        plan = resolve(
            "cast('compute')",
            input_shape=("B", 12),
            output_shape=("B", 12),
            dtype=dtype,
            input_dtype="int64",
        )
        model = build(plan, device=device)
        x = torch.randint(0, 10, (3, 12), device=device, dtype=torch.int64)
        out = model(x=x)["output"]
        assert out.dtype == getattr(torch, dtype)
        assert out.shape == (3, 12)


def test_propagates_int64_into_embedding_layer(device):
    plan = resolve("cast('int64')\nembedding(100, 32)", input_shape=("B", 16), output_shape=("B", 16, 32))
    model = build(plan, device=device)
    x = torch.randint(0, 100, (2, 16), device=device).float()
    out = model(x=x)["output"]
    assert out.shape == (2, 16, 32)
    assert out.dtype == torch.float32


def test_rejects_invalid_dtype():
    with pytest.raises(HNDLError, match="E_ARGUMENT.*dtype"):
        resolve("cast('complex128')", input_shape=("B", 8), output_shape=("B", 8))


def test_preserves_tensor_ranks_2_3_4(device):
    for shape in [(2, 16), (2, 5, 16), (2, 3, 8, 8)]:
        plan = resolve("cast('float32')", input_shape=("B", *shape[1:]), output_shape=("B", *shape[1:]))
        model = build(plan, device=device)
        x = torch.randn(*shape, device=device)
        out = model(x=x)["output"]
        assert out.shape == shape


def test_gradcheck_double(device):
    plan = resolve("cast('float64')", input_shape=("B", 4), output_shape=("B", 4), dtype="float32")
    model = build(plan, device=device)
    cast_module = model[plan.nodes[0].id]
    x = torch.randn(2, 4, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(cast_module, (x,), eps=1e-6, atol=1e-4)
