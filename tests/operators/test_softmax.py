"""Numerics, dim validation, and inference for ``softmax``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build

CASES = [(("B", 16), -1), (("B", 16), 1), (("B", 4, 8), -1), (("B", 4, 8), 1), (("B", 4, 8), 2),
         (("B", 2, 5, 5), -1), (("B", 2, 5, 5), 1), (("B", 2, 5, 5), -3)]


def pair(shape, device):
    x = torch.randn(3, *shape[1:], device=device)
    return x.clone().requires_grad_(), x.clone().requires_grad_()


@pytest.mark.parametrize("shape,dim", CASES)
def test_matches_torch_functional_forward_and_backward(shape, dim, device):
    plan = resolve(f"softmax({dim})", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    x, mirror = pair(shape, device)
    actual = model(x=x)["output"]
    expected = F.softmax(mirror, dim=dim)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


@pytest.mark.parametrize("shape,dim", CASES)
def test_slices_sum_to_one_and_stay_in_the_unit_interval(shape, dim, device):
    plan = resolve(f"softmax({dim})", input_shape=shape, output_shape=shape)
    model = build(plan, device=device)
    out = model(x=torch.randn(3, *shape[1:], device=device) * 2)["output"]
    totals = out.sum(dim=dim)
    torch.testing.assert_close(totals, torch.ones_like(totals))
    assert ((out > 0) & (out < 1)).all()


def test_negative_dims_agree_with_their_positive_counterparts(device):
    shape = ("B", 3, 4, 4)
    negative = build(resolve("softmax(-3)", input_shape=shape, output_shape=shape), device=device)
    positive = build(resolve("softmax(1)", input_shape=shape, output_shape=shape), device=device)
    x = torch.randn(2, 3, 4, 4, device=device)
    torch.testing.assert_close(negative(x=x)["output"], positive(x=x)["output"])


@pytest.mark.parametrize("dim", [0, -2, -3])
def test_batch_axis_is_rejected(dim):
    with pytest.raises(HNDLError) as excinfo:
        resolve(f"softmax({dim})", input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"


@pytest.mark.parametrize("shape,dim", [(("B", 8), 2), (("B", 8), -3), (("B", 4, 8), 3), (("B", 4, 8), -4),
                                       (("B", 2, 5, 5), 4)])
def test_out_of_range_dims_are_rejected(shape, dim):
    with pytest.raises(HNDLError) as excinfo:
        resolve(f"softmax({dim})", input_shape=shape, output_shape=shape)
    assert excinfo.value.code == "E_ARGUMENT"


@pytest.mark.parametrize("source", ['softmax("last")', "softmax(dim=-1, axis=2)"])
def test_bad_arguments_report_e_argument(source):
    with pytest.raises(HNDLError) as excinfo:
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))
    assert excinfo.value.code == "E_ARGUMENT"


def test_rank_dependent_dim_is_checked_against_the_inferred_rank():
    """dim=3 is fine once the incoming tensor is known to be rank 4."""
    plan = resolve("conv(4, kernel_size=1)\nsoftmax(3)", input_shape=("B", 3, 5, 5), output_shape=("B", 4, 5, 5))
    assert plan.nodes[1].args["dim"] == 3
    with pytest.raises(HNDLError) as excinfo:
        resolve("linear(4)\nsoftmax(3)", input_shape=("B", 8), output_shape=("B", 4))
    assert excinfo.value.code == "E_ARGUMENT"


def test_shapes_flow_backward_through_the_activation():
    plan = resolve("linear()\nsoftmax()", input_shape=("B", 32), output_shape=("B", 7))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7)
    assert plan.nodes[1].input_shapes["x"] == ("B", 7)


def test_shapes_flow_forward_through_the_activation():
    plan = resolve("softmax(1)\nlinear()", input_shape=("B", 3, 6), output_shape=("B", 3, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_rows_still_sum_to_one(dtype):
    shape = ("B", 4, 16)
    plan = resolve("softmax()", input_shape=shape, output_shape=shape, dtype=dtype)
    model = build(plan, device="cuda:0")
    x = torch.randn(3, 4, 16, device="cuda:0", dtype=DTYPES[dtype]) * 8
    out = model(x=x)["output"]
    assert out.dtype == DTYPES[dtype]
    tolerance = {"float16": 5e-3, "bfloat16": 6e-2}[dtype]
    torch.testing.assert_close(out.float().sum(dim=-1), torch.ones(3, 4, device="cuda:0"),
                               atol=tolerance, rtol=0)
    torch.testing.assert_close(out.float(), F.softmax(x.float(), dim=-1), atol=tolerance, rtol=0)
