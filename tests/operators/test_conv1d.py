"""Numerics, shape arithmetic, and error codes for the one-dimensional convolution."""

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build


COMBINATIONS = [
    # kernel_size, stride, padding, dilation, groups, length
    (1, 1, 0, 1, 1, 16),
    (3, 1, 1, 1, 1, 16),
    (3, 2, 1, 1, 1, 15),
    (4, 2, 1, 1, 1, 32),
    (3, 1, 0, 2, 1, 16),
    (5, 3, 2, 2, 1, 20),
    (3, 1, 1, 1, 2, 9),
]


def conv1d_source(out_channels, kernel_size, stride, padding, dilation, groups, bias=True):
    return (f"conv1d({out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, "
            f"dilation={dilation}, groups={groups}, bias={bias})")


def torch_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)


@pytest.mark.parametrize("kernel_size,stride,padding,dilation,groups,length", COMBINATIONS)
def test_forward_length_arithmetic_agrees_with_torch(kernel_size, stride, padding, dilation, groups, length):
    in_channels, out_channels = 4 * groups, 6 * groups
    reference = torch_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
    expected = reference(torch.zeros(1, in_channels, length)).shape[2]
    source = conv1d_source(out_channels, kernel_size, stride, padding, dilation, groups)
    plan = resolve(source, input_shape=("B", in_channels, length), output_shape=("B", out_channels, expected))
    assert plan.nodes[0].output_shapes["out"] == ("B", out_channels, expected)
    model = build(plan, device="cpu")
    assert model(x=torch.randn(2, in_channels, length))["output"].shape == (2, out_channels, expected)


@pytest.mark.parametrize("kernel_size,stride,padding,dilation,groups,length", COMBINATIONS)
def test_parity_with_nn_conv1d_forward_and_gradients(kernel_size, stride, padding, dilation, groups, length, device):
    torch.manual_seed(kernel_size * 100 + stride * 10 + padding)
    in_channels, out_channels = 4 * groups, 6 * groups
    reference = torch_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups).to(device)
    expected_length = reference(torch.zeros(1, in_channels, length, device=device)).shape[2]
    source = conv1d_source(out_channels, kernel_size, stride, padding, dilation, groups)
    plan = resolve(source, input_shape=("B", in_channels, length),
                   output_shape=("B", out_channels, expected_length))
    model = build(plan, device=device, initialization_seed=11)
    module = model[plan.nodes[0].id]
    assert isinstance(module, nn.Conv1d)
    reference.load_state_dict(module.state_dict())

    x = torch.randn(3, in_channels, length, device=device)
    actual_input = x.clone().requires_grad_()
    expected_input = x.clone().requires_grad_()
    actual = model(x=actual_input)["output"]
    expected = reference(expected_input)
    torch.testing.assert_close(actual, expected)

    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(module.weight.grad, reference.weight.grad)
    torch.testing.assert_close(module.bias.grad, reference.bias.grad)


def test_bias_free_and_depthwise_parity(device):
    reference = torch_conv(8, 8, 3, 1, 1, 1, groups=8, bias=False).to(device)
    plan = resolve(conv1d_source(8, 3, 1, 1, 1, 8, bias=False),
                   input_shape=("B", 8, 12), output_shape=("B", 8, 12))
    model = build(plan, device=device, initialization_seed=2)
    module = model[plan.nodes[0].id]
    assert module.bias is None and module.weight.shape == (8, 1, 3)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(2, 8, 12, device=device)
    torch.testing.assert_close(model(x=x)["output"], reference(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_matches_nn_conv1d_on_cuda(dtype):
    plan = resolve(conv1d_source(6, 3, 2, 1, 1, 1), input_shape=("B", 4, 17),
                   output_shape=("B", 6, 9), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=7)
    module = model[plan.nodes[0].id]
    assert module.weight.dtype == DTYPES[dtype]
    reference = torch_conv(4, 6, 3, 2, 1, 1, 1).to("cuda:0", DTYPES[dtype])
    reference.load_state_dict(module.state_dict())
    x = torch.randn(2, 4, 17, device="cuda:0", dtype=DTYPES[dtype])
    output = model(x=x)["output"]
    assert output.dtype == DTYPES[dtype]
    torch.testing.assert_close(output, reference(x))


def test_in_channels_and_out_channels_are_inferred_in_both_directions():
    plan = resolve("conv1d(kernel_size=3, padding=1)", input_shape=("B", 3, 20), output_shape=("B", 9, 20))
    node = plan.nodes[0]
    assert node.args["in_channels"] == 3 and node.args["out_channels"] == 9
    assert node.provenance["in_channels"] == "inferred" and node.provenance["out_channels"] == "inferred"
    assert node.provenance["stride"] == "operator default"
    model = build(plan, device="cpu")
    assert model[node.id].weight.shape == (9, 3, 3)


def test_input_length_is_inferred_backward_when_the_inverse_is_unique():
    plan = resolve("linear()\nreshape(2)\nconv1d(4, kernel_size=3, padding=1)",
                   input_shape=("B", 16), output_shape=("B", 4, 8))
    assert plan.nodes[0].args["out_features"] == 16
    assert plan.nodes[1].output_shapes["out"] == ("B", 2, 8)
    assert plan.nodes[2].args["in_channels"] == 2
    model = build(plan, device="cpu")
    assert model(x=torch.randn(2, 16))["output"].shape == (2, 4, 8)


def test_strided_inverse_leaves_an_interval_and_is_reported_as_ambiguous():
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("linear()\nreshape(1)\nconv1d(1, kernel_size=3, stride=2, padding=1)",
                input_shape=("B", 16), output_shape=("B", 1, 4))


def test_inverse_without_a_positive_input_length_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("linear()\nreshape(1)\nconv1d(1, kernel_size=1, padding=3)",
                input_shape=("B", 16), output_shape=("B", 1, 1))


@pytest.mark.parametrize("in_channels,out_channels", [(8, 6), (6, 8)])
def test_groups_must_divide_both_channel_counts(in_channels, out_channels):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*must divide"):
        resolve(conv1d_source(out_channels, 1, 1, 0, 1, 4),
                input_shape=("B", in_channels, 8), output_shape=("B", out_channels, 8))


def test_contradictory_output_length_is_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(conv1d_source(4, 3, 1, 1, 1, 1), input_shape=("B", 2, 16), output_shape=("B", 4, 15))


@pytest.mark.parametrize("source,message", [
    ("conv1d(4, kernel_size=0)", "E_ARGUMENT"),
    ("conv1d(4, kernel_size=3, stride=0)", "E_ARGUMENT"),
    ("conv1d(4, kernel_size=3, padding=-1)", "E_ARGUMENT"),
    ("conv1d(4, kernel_size=3, dilation=0)", "E_ARGUMENT"),
    ("conv1d(4)", "kernel_size"),
])
def test_invalid_scalar_arguments_are_rejected(source, message):
    with pytest.raises(HNDLError, match=message):
        resolve(source, input_shape=("B", 2, 16), output_shape=("B", 4, 16))


def test_rank_two_and_rank_four_inputs_are_rejected():
    for input_shape, output_shape in ((("B", 16), ("B", 4, 16)), (("B", 2, 8, 8), ("B", 4, 8, 8))):
        with pytest.raises(HNDLError, match="E_CONSTRAINT"):
            resolve(conv1d_source(4, 3, 1, 1, 1, 1), input_shape=input_shape, output_shape=output_shape)
