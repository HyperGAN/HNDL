"""transpose: parity with ``torch.transpose``, bidirectional inference, error codes."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build


def swapped(shape, dim0, dim1):
    result = list(shape)
    result[dim0], result[dim1] = shape[dim1], shape[dim0]
    return tuple(result)


def compiled(source, input_shape, output_shape, device):
    return build(resolve(source, input_shape=input_shape, output_shape=output_shape), device=device)


@pytest.mark.parametrize("shape,dim0,dim1", [
    ((3, 7), 1, 1),
    ((3, 5, 7), 1, 2),
    ((3, 5, 7), 2, 1),
    ((3, 5, 7), 2, 2),
    ((2, 3, 5, 7), 1, 2),
    ((2, 3, 5, 7), 1, 3),
    ((2, 3, 5, 7), 2, 3),
    ((2, 3, 5, 7), 3, 1),
])
def test_forward_and_gradients_match_torch_transpose(shape, dim0, dim1, device):
    target = swapped(shape, dim0, dim1)
    model = compiled(f"transpose({dim0}, {dim1})", ("B", *shape[1:]), ("B", *target[1:]), device)
    x = torch.randn(*shape, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    weight = torch.randn(*target, device=device)
    output = model(x=x)["output"]
    expected = torch.transpose(mirror, dim0, dim1)
    assert tuple(output.shape) == target
    torch.testing.assert_close(output, expected)
    (output * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_two_swaps_of_the_same_pair_restore_the_original(device):
    model = compiled("transpose(1, 3)\ntranspose(1, 3)", ("B", 3, 5, 7), ("B", 3, 5, 7), device)
    x = torch.randn(4, 3, 5, 7, device=device)
    torch.testing.assert_close(model(x=x)["output"], x)


def test_output_contract_determines_the_input_axes():
    plan = resolve("linear()\ntranspose(1, 2)", input_shape=("B", 4, 6), output_shape=("B", 10, 4))
    assert plan.nodes[0].args["out_features"] == 10
    assert plan.nodes[1].input_shapes["x"] == ("B", 4, 10)
    assert plan.nodes[1].output_shapes["out"] == ("B", 10, 4)


def test_backward_inference_through_a_rank_four_swap():
    plan = resolve("conv(kernel_size=3, padding=1)\ntranspose(1, 3)", input_shape=("B", 3, 8, 8),
                   output_shape=("B", 8, 8, 5))
    assert plan.nodes[0].args["out_channels"] == 5
    assert plan.nodes[1].input_shapes["x"] == ("B", 5, 8, 8)


def test_sequence_and_channel_layouts_bridge_in_both_directions():
    plan = resolve("transpose(1, 2)\ntranspose(1, 2)\nlinear()", input_shape=("B", 6, 4), output_shape=("B", 6, 9))
    assert [node.output_shapes["out"] for node in plan.nodes] == [("B", 4, 6), ("B", 6, 4), ("B", 6, 9)]


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("transpose(0, 1)", ("B", 4, 6), ("B", 4, 6), "E_ARGUMENT"),
    ("transpose(1, 0)", ("B", 4, 6), ("B", 4, 6), "E_ARGUMENT"),
    ("transpose(1, 4)", ("B", 4, 6), ("B", 4, 6), "E_ARGUMENT"),
    ("transpose(1, 2)", ("B", 8), ("B", 8), "E_ARGUMENT"),
    ("transpose(2, 3)", ("B", 4, 6), ("B", 4, 6), "E_ARGUMENT"),
    ("transpose(1)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("transpose(1, 2)", ("B", 4, 6), ("B", 4, 6), "E_CONSTRAINT"),
])
def test_invalid_axes_and_contradictions_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_identity_swap_is_accepted_and_keeps_the_shape(device):
    model = compiled("transpose(2, 2)", ("B", 4, 6), ("B", 4, 6), device)
    x = torch.randn(2, 4, 6, device=device)
    torch.testing.assert_close(model(x=x)["output"], x)
