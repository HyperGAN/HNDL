"""The ``conv`` operator's "down2" policy: filled arguments, exact halving, numerics."""

import pytest
import torch
from torch import nn

from hndl import resolve
from hndl.errors import HNDLError
from hndl.torch import DTYPES, build

DOWN2 = "spatial.down2@1"
REQUIRED = {"kernel_size": (4, 4), "stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "groups": 1}


def test_policy_fills_the_downsampling_arguments_and_records_their_origin():
    plan = resolve('conv(8, policy="down2")', input_shape=("B", 3, 16, 16), output_shape=("B", 8, 8, 8))
    node = plan.nodes[0]
    assert node.source["policy"] == DOWN2
    for name, value in REQUIRED.items():
        assert node.args[name] == value
        assert node.provenance[name] == "policy-selected"
    assert node.provenance["out_channels"] == "explicit"
    assert node.provenance["bias"] == "operator default"


def test_policy_agrees_with_explicit_arguments_but_rejects_contradictions():
    plan = resolve('conv(8, policy="down2", kernel_size=4, bias=False)',
                   input_shape=("B", 3, 16, 16), output_shape=("B", 8, 8, 8))
    assert plan.nodes[0].args["kernel_size"] == (4, 4) and plan.nodes[0].args["bias"] is False
    assert plan.nodes[0].provenance["kernel_size"] == "explicit"
    for conflicting in ("stride=1", "padding=0", "kernel_size=3", "dilation=2", "groups=2"):
        with pytest.raises(HNDLError, match="E_POLICY_CONFLICT"):
            resolve(f'conv(8, policy="down2", {conflicting})',
                    input_shape=("B", 4, 16, 16), output_shape=("B", 8, 8, 8))


@pytest.mark.parametrize("height,width", [(16, 16), (32, 8), (6, 10)])
def test_forward_inference_halves_each_spatial_axis(height, width):
    plan = resolve('conv(8, policy="down2")', input_shape=("B", 3, height, width),
                   output_shape=("B", 8, height // 2, width // 2))
    assert plan.nodes[0].output_shapes["out"] == ("B", 8, height // 2, width // 2)


def test_backward_inference_is_unique_through_a_stack_of_policy_convolutions():
    """Without the policy this is the classic off-by-one interval, so it stays ambiguous."""
    source = 'linear()\nreshape(32)\nconv(16, policy="down2")\nconv(3, policy="down2")'
    plan = resolve(source, input_shape=("B", 16), output_shape=("B", 3, 8, 8))
    assert plan.nodes[1].args["shape"] == (32, 32, 32)
    assert plan.nodes[0].args["out_features"] == 32 * 32 * 32
    assert plan.nodes[2].input_shapes["x"] == ("B", 32, 32, 32)
    assert plan.nodes[2].output_shapes["out"] == ("B", 16, 16, 16)
    assert plan.nodes[3].output_shapes["out"] == ("B", 3, 8, 8)
    ambiguous = 'linear()\nreshape(32)\nconv(16, kernel_size=4, stride=2, padding=1)'
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve(ambiguous, input_shape=("B", 16), output_shape=("B", 16, 16, 16))


@pytest.mark.parametrize("shape", [("B", 3, 9, 8), ("B", 3, 8, 9), ("B", 3, 7, 7)])
def test_odd_input_extents_are_rejected_under_the_policy(shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve('conv(8, policy="down2")', input_shape=shape,
                output_shape=("B", 8, shape[2] // 2, shape[3] // 2))
    # The same convolution without the policy accepts the odd extent.
    plan = resolve("conv(8, kernel_size=4, stride=2, padding=1)", input_shape=shape,
                   output_shape=("B", 8, shape[2] // 2, shape[3] // 2))
    assert plan.nodes[0].output_shapes["out"] == ("B", 8, shape[2] // 2, shape[3] // 2)


def test_unknown_policy_name_is_reported():
    with pytest.raises(HNDLError, match="E_POLICY_CONFLICT"):
        resolve('conv(8, policy="up2")', input_shape=("B", 3, 16, 16), output_shape=("B", 8, 8, 8))


@pytest.mark.parametrize("channels,height,width", [(3, 16, 16), (6, 12, 20)])
def test_forward_matches_nn_conv2d_with_kernel_four_stride_two_padding_one(device, channels, height, width):
    plan = resolve('conv(8, policy="down2")', input_shape=("B", channels, height, width),
                   output_shape=("B", 8, height // 2, width // 2))
    model = build(plan, device=device, initialization_seed=11)
    module = model[plan.nodes[0].id]
    reference = nn.Conv2d(channels, 8, 4, stride=2, padding=1).to(device)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(4, channels, height, width, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual, expected = module(x), reference(mirror)
    assert tuple(actual.shape) == (4, 8, height // 2, width // 2)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_discriminator_stack_runs_in_reduced_precision(dtype):
    source = 'conv(16, policy="down2")\nleaky_relu(0.2)\nconv(32, policy="down2")\nflatten()\nlinear()'
    plan = resolve(source, input_shape=("B", 3, 16, 16), output_shape=("B", 1), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=5)
    x = torch.randn(2, 3, 16, 16, device="cuda:0", dtype=DTYPES[dtype])
    output = model(x=x)["output"]
    assert output.dtype == DTYPES[dtype] and tuple(output.shape) == (2, 1)
    assert output.isfinite().all()
