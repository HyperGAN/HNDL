"""Numerics, inference and error codes for the ``l2_norm`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "l2_norm")


@pytest.mark.parametrize("shape", [(4, 16), (3, 5, 24), (2, 4, 8, 8)])
def test_output_has_unit_l2_norm_along_last_axis(device, shape):
    plan = resolve("l2_norm()", input_shape=("B", *shape[1:]), output_shape=("B", *shape[1:]))
    model = build(plan, device=device)
    x = torch.randn(*shape, device=device) * 5.0 + 1.0
    out = model(x=x)["output"]
    norm = torch.linalg.vector_norm(out, ord=2, dim=-1)
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim", [-1, 1, 2])
def test_supports_different_normalization_axes(device, dim):
    shape = (3, 6, 12)
    plan = resolve(f"l2_norm(dim={dim})", input_shape=("B", 6, 12), output_shape=("B", 6, 12))
    model = build(plan, device=device)
    x = torch.randn(*shape, device=device)
    out = model(x=x)["output"]
    norm = torch.linalg.vector_norm(out, ord=2, dim=dim)
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5)


def test_matches_functional_normalize_including_gradients(device):
    plan = resolve("l2_norm(eps=1e-5)", input_shape=("B", 6, 20), output_shape=("B", 6, 20))
    model = build(plan, device=device)
    x = torch.randn(3, 6, 20, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.normalize(mirror, p=2.0, dim=-1, eps=1e-5)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_non_positive_eps_is_an_argument_error(eps):
    with pytest.raises(HNDLError, match="E_ARGUMENT.*eps"):
        resolve(f"l2_norm(eps={eps})", input_shape=("B", 8), output_shape=("B", 8))


def test_has_no_parameters(device):
    plan = resolve("l2_norm()", input_shape=("B", 16), output_shape=("B", 16))
    model = build(plan, device=device)
    assert list(model.parameters()) == []


def test_width_flows_forward_and_backward():
    forward = resolve("l2_norm()\nlinear()", input_shape=("B", 16), output_shape=("B", 4))
    assert forward.nodes[0].output_shapes["out"] == ("B", 16)
    assert forward.nodes[1].args["in_features"] == 16
    backward = resolve("linear()\nl2_norm()", input_shape=("B", 8), output_shape=("B", 32))
    assert backward.nodes[0].args["out_features"] == 32


def test_gradcheck_double(device):
    plan = resolve("l2_norm(eps=1e-5)", input_shape=("B", 4, 8), output_shape=("B", 4, 8))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    x = torch.randn(2, 4, 8, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve("l2_norm()", input_shape=("B", 5, 16), output_shape=("B", 5, 16), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 5, 16, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
