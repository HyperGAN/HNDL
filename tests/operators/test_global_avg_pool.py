"""Numerics, bidirectional inference, and error codes for ``global_avg_pool``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build


@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (1, 7, 5, 11), (4, 16, 1, 1)])
def test_matches_the_spatial_mean_forward_and_backward(shape, device):
    batch, channels, height, width = shape
    plan = resolve("global_avg_pool()", input_shape=("B", channels, height, width),
                   output_shape=("B", channels))
    model = build(plan, device=device)
    x = torch.randn(*shape, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = mirror.mean(dim=(2, 3))
    assert tuple(actual.shape) == (batch, channels)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)
    # The gradient is the uniform 1/(H*W) spread of the upstream gradient.
    torch.testing.assert_close(x.grad, (2 * expected / (height * width)).detach()[:, :, None, None]
                               .expand_as(x).contiguous())


def test_matches_adaptive_avg_pool_one_followed_by_flatten(device):
    source = "conv(6, kernel_size=3, padding=1)\nglobal_avg_pool()"
    equivalent = "conv(6, kernel_size=3, padding=1)\nadaptive_avg_pool(1)\nflatten()"
    kwargs = dict(input_shape=("B", 3, 9, 9), output_shape=("B", 6))
    pooled = build(resolve(source, **kwargs), device=device, initialization_seed=11)
    staged = build(resolve(equivalent, **kwargs), device=device, initialization_seed=11)
    x = torch.randn(2, 3, 9, 9, device=device)
    torch.testing.assert_close(pooled(x=x)["output"], staged(x=x)["output"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_stays_in_the_plan_dtype(dtype):
    plan = resolve("global_avg_pool()", input_shape=("B", 8, 6, 6), output_shape=("B", 8), dtype=dtype)
    model = build(plan, device="cuda:0")
    x = torch.randn(2, 8, 6, 6, device="cuda:0", dtype=DTYPES[dtype])
    out = model(x=x)["output"]
    assert out.dtype == DTYPES[dtype]
    torch.testing.assert_close(out.float(), x.float().mean(dim=(2, 3)), rtol=2e-3, atol=2e-3)


def test_a_following_linear_infers_in_features_from_the_channel_count():
    plan = resolve("conv(24, kernel_size=3, padding=1)\nglobal_avg_pool()\nlinear()",
                   input_shape=("B", 3, 16, 16), output_shape=("B", 10))
    assert plan.nodes[1].output_shapes["out"] == ("B", 24)
    assert plan.nodes[2].args["in_features"] == 24 and plan.nodes[2].args["out_features"] == 10


def test_channels_flow_backward_through_the_pool():
    plan = resolve("conv(kernel_size=3, padding=1)\nglobal_avg_pool()",
                   input_shape=("B", 3, 8, 8), output_shape=("B", 9))
    assert plan.nodes[0].args["out_channels"] == 9
    assert plan.nodes[1].input_shapes["x"] == ("B", 9, 8, 8)


def test_spatial_extents_are_not_inferable_backward():
    """The pool erases H and W, so nothing downstream can recover them."""
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("reshape()\nglobal_avg_pool()", input_shape=("B", 48), output_shape=("B", 3))


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("global_avg_pool()", ("B", 16), ("B", 16), "E_CONSTRAINT"),
    ("global_avg_pool()", ("B", 4, 16), ("B", 4), "E_CONSTRAINT"),
    ("global_avg_pool()", ("B", 4, 8, 8), ("B", 5), "E_CONSTRAINT"),
    ("global_avg_pool()", ("B", 4, 8, 8), ("B", 4, 1, 1), "E_CONSTRAINT"),
    ("global_avg_pool(2)", ("B", 4, 8, 8), ("B", 4), "E_ARGUMENT"),
])
def test_invalid_shapes_and_arguments_report_their_codes(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_the_pool_has_no_parameters_and_is_stateless(device):
    plan = resolve("global_avg_pool()", input_shape=("B", 4, 5, 5), output_shape=("B", 4))
    model = build(plan, device=device)
    module = model[plan.nodes[0].id]
    assert list(module.parameters()) == [] and list(module.buffers()) == []
    x = torch.randn(2, 4, 5, 5, device=device)
    model.train()
    trained = model(x=x)["output"]
    model.eval()
    torch.testing.assert_close(model(x=x)["output"], trained)
