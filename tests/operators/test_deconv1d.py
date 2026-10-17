"""Numerics, inference and error codes for the ``deconv1d`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, Registry, resolve
from hndl.testing import check_operator
from hndl.torch import DTYPES, build, parameter_counts

REGISTRY = Registry.builtins()


def test_passes_standard_operator_checks():
    check_operator(REGISTRY, "deconv1d")


def test_up2_policy_doubles_length_exactly(device):
    plan = resolve('deconv1d(8, policy="up2")', input_shape=("B", 4, 16), output_shape=("B", 8, 32))
    model = build(plan, device=device)
    x = torch.randn(2, 4, 16, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 32)


def test_length_flows_bidirectionally():
    forward = resolve('deconv1d(16, policy="up2")', input_shape=("B", 8, 20), output_shape=("B", 16, 40))
    assert forward.nodes[0].output_shapes["out"] == ("B", 16, 40)

    backward = resolve('deconv1d(policy="up2")', input_shape=("B", 8, 15), output_shape=("B", 16, 30))
    assert backward.nodes[0].args["out_channels"] == 16
    assert backward.nodes[0].input_shapes["x"] == ("B", 8, 15)


def test_parameter_counts():
    plan = resolve("deconv1d(16, kernel_size=5, bias=True)", input_shape=("B", 8, 10), output_shape=("B", 16, 14))
    # in_channels * out_channels * kernel_size + out_channels = 8 * 16 * 5 + 16 = 656
    assert parameter_counts(plan) == {"n0": 656}


def test_matches_functional_conv_transpose1d(device):
    plan = resolve("deconv1d(6, kernel_size=3, stride=2, padding=1)", input_shape=("B", 4, 10), output_shape=("B", 6, 19))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    x = torch.randn(2, 4, 10, device=device)
    actual = model(x=x)["output"]
    expected = F.conv_transpose1d(x, module.weight, module.bias, stride=2, padding=1)
    torch.testing.assert_close(actual, expected)


def test_spectral_norm_option(device):
    plan = resolve('deconv1d(8, policy="up2", spectral_norm=True)', input_shape=("B", 4, 12), output_shape=("B", 8, 24))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    assert hasattr(module, "parametrizations")
    x = torch.randn(2, 4, 12, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 24)
    assert out.isfinite().all()


def test_invalid_output_padding_reports_e_argument():
    with pytest.raises(HNDLError, match="E_ARGUMENT.*output_padding"):
        resolve("deconv1d(8, kernel_size=3, stride=1, dilation=1, output_padding=1)",
                input_shape=("B", 4, 10), output_shape=("B", 8, 12))


def test_groups_divisibility_reports_e_constraint():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*groups"):
        resolve("deconv1d(7, kernel_size=3, groups=2)",
                input_shape=("B", 4, 10), output_shape=("B", 7, 12))


def test_rejects_rank_2_and_rank_4():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve('deconv1d(8, policy="up2")', input_shape=("B", 16), output_shape=("B", 8))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve('deconv1d(8, policy="up2")', input_shape=("B", 4, 16, 16), output_shape=("B", 8, 32, 32))


def test_gradcheck_double(device):
    plan = resolve("deconv1d(4, kernel_size=3, stride=1, padding=1)", input_shape=("B", 2, 8), output_shape=("B", 4, 8))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id].double()
    x = torch.randn(2, 2, 8, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_on_cuda(dtype):
    plan = resolve('deconv1d(8, policy="up2")', input_shape=("B", 4, 16), output_shape=("B", 8, 32), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 4, 16, device="cuda:0", dtype=torch_dtype)
    out = model(x=x)["output"]
    assert out.dtype == torch_dtype
    assert out.isfinite().all()
