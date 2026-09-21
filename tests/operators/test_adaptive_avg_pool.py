"""Numerics, bidirectional inference, and error codes for ``adaptive_avg_pool``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build


def pool(source, input_shape, output_shape, device):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    return plan, build(plan, device=device)


@pytest.mark.parametrize("spatial,output_size", [
    ((28, 28), 1),
    ((15, 15), 4),
    ((9, 6), (3, 2)),
    ((7, 7), 7),
    ((5, 5), 8),
    ((4, 6), (8, 3)),
])
def test_matches_torch_forward_and_backward(spatial, output_size, device):
    height, width = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    source = f"adaptive_avg_pool({output_size})"
    _, model = pool(source, ("B", 5, *spatial), ("B", 5, height, width), device)
    x = torch.randn(3, 5, *spatial, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]
    expected = F.adaptive_avg_pool2d(mirror, (height, width))
    assert tuple(actual.shape) == (3, 5, height, width)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_output_size_one_is_the_spatial_mean(device):
    _, model = pool("adaptive_avg_pool(1)", ("B", 4, 11, 13), ("B", 4, 1, 1), device)
    x = torch.randn(2, 4, 11, 13, device=device)
    torch.testing.assert_close(model(x=x)["output"], x.mean(dim=(2, 3), keepdim=True))


def test_output_larger_than_input_replicates_like_torch(device):
    """torch permits an output grid larger than the input; windows then repeat elements."""
    _, model = pool("adaptive_avg_pool(6)", ("B", 2, 3, 3), ("B", 2, 6, 6), device)
    x = torch.randn(2, 2, 3, 3, device=device)
    out = model(x=x)["output"]
    torch.testing.assert_close(out, F.adaptive_avg_pool2d(x, (6, 6)))
    torch.testing.assert_close(out[:, :, 0, 0], x[:, :, 0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_stays_in_the_plan_dtype(dtype):
    plan = resolve("adaptive_avg_pool(2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 2, 2), dtype=dtype)
    model = build(plan, device="cuda:0")
    x = torch.randn(2, 3, 8, 8, device="cuda:0", dtype=DTYPES[dtype])
    out = model(x=x)["output"]
    assert out.dtype == DTYPES[dtype]
    torch.testing.assert_close(out.float(), F.adaptive_avg_pool2d(x.float(), (2, 2)), rtol=2e-3, atol=2e-3)


def test_output_size_is_inferred_backward_from_the_output_contract():
    plan = resolve("adaptive_avg_pool()", input_shape=("B", 6, 9, 9), output_shape=("B", 6, 3, 3))
    node = plan.nodes[0]
    assert node.args["output_size"] == (3, 3)
    assert node.provenance["output_size"] == "inferred"
    assert node.output_shapes["out"] == ("B", 6, 3, 3)


def test_an_int_output_size_normalizes_to_a_pair_and_a_pair_is_kept():
    square = resolve("adaptive_avg_pool(2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 2, 2))
    assert square.nodes[0].args["output_size"] == (2, 2)
    rectangle = resolve("adaptive_avg_pool((2, 4))", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 2, 4))
    assert rectangle.nodes[0].args["output_size"] == (2, 4)


def test_channels_flow_backward_through_the_pool():
    plan = resolve("conv(kernel_size=3, padding=1)\nadaptive_avg_pool(2)",
                   input_shape=("B", 3, 8, 8), output_shape=("B", 12, 2, 2))
    assert plan.nodes[0].args["out_channels"] == 12
    assert plan.nodes[1].input_shapes["x"] == ("B", 12, 8, 8)


def test_input_spatial_extents_are_not_inferable_backward():
    """Every input extent maps to the requested output, so the inverse is unconstrained."""
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("reshape()\nadaptive_avg_pool(2)", input_shape=("B", 48), output_shape=("B", 3, 2, 2))


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("adaptive_avg_pool(0)", ("B", 3, 8, 8), ("B", 3, 1, 1), "E_ARGUMENT"),
    ("adaptive_avg_pool(-2)", ("B", 3, 8, 8), ("B", 3, 1, 1), "E_ARGUMENT"),
    ("adaptive_avg_pool((1, 2, 3))", ("B", 3, 8, 8), ("B", 3, 1, 1), "E_ARGUMENT"),
    ("adaptive_avg_pool(2.5)", ("B", 3, 8, 8), ("B", 3, 2, 2), "E_ARGUMENT"),
    ("adaptive_avg_pool(2)", ("B", 3, 8, 8), ("B", 3, 4, 4), "E_CONSTRAINT"),
    ("adaptive_avg_pool(2)", ("B", 16), ("B", 16), "E_CONSTRAINT"),
    ("adaptive_avg_pool(2)", ("B", 4, 8, 8), ("B", 4, 2), "E_CONSTRAINT"),
])
def test_invalid_arguments_and_shapes_report_their_codes(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_the_pool_has_no_parameters_and_is_stateless(device):
    plan, model = pool("adaptive_avg_pool(3)", ("B", 4, 9, 9), ("B", 4, 3, 3), device)
    module = model[plan.nodes[0].id]
    assert list(module.parameters()) == [] and list(module.buffers()) == []
    x = torch.randn(2, 4, 9, 9, device=device)
    model.train()
    trained = model(x=x)["output"]
    model.eval()
    torch.testing.assert_close(model(x=x)["output"], trained)
