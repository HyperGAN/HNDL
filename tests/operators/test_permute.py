"""permute: parity with ``torch.permute``, bidirectional inference, error codes."""

from itertools import permutations

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import build


def reordered(shape, dims):
    return (shape[0], *(shape[dim] for dim in dims))


def compiled(source, input_shape, output_shape, device):
    return build(resolve(source, input_shape=input_shape, output_shape=output_shape), device=device)


def cases():
    for shape in ((3, 7), (3, 5, 7), (2, 3, 5, 7)):
        for dims in permutations(range(1, len(shape))):
            yield pytest.param(shape, dims, id=f"{len(shape)}d-{''.join(map(str, dims))}")


@pytest.mark.parametrize("shape,dims", list(cases()))
def test_forward_and_gradients_match_torch_permute(shape, dims, device):
    target = reordered(shape, dims)
    source = f"permute({', '.join(map(str, dims))})"
    model = compiled(source, ("B", *shape[1:]), ("B", *target[1:]), device)
    x = torch.randn(*shape, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    weight = torch.randn(*target, device=device)
    output = model(x=x)["output"]
    expected = mirror.permute((0, *dims))
    assert tuple(output.shape) == target
    torch.testing.assert_close(output, expected)
    (output * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_a_permutation_and_its_inverse_restore_the_original(device):
    model = compiled("permute(3, 1, 2)\npermute(2, 3, 1)", ("B", 3, 5, 7), ("B", 3, 5, 7), device)
    x = torch.randn(4, 3, 5, 7, device=device)
    torch.testing.assert_close(model(x=x)["output"], x)


def test_a_two_axis_permute_agrees_with_the_matching_transpose(device):
    x = torch.randn(4, 5, 7, device=device)
    by_permute = compiled("permute(2, 1)", ("B", 5, 7), ("B", 7, 5), device)
    by_transpose = compiled("transpose(1, 2)", ("B", 5, 7), ("B", 7, 5), device)
    torch.testing.assert_close(by_permute(x=x)["output"], by_transpose(x=x)["output"])


def test_output_contract_determines_the_input_axes():
    plan = resolve("linear()\npermute(2, 1)", input_shape=("B", 4, 6), output_shape=("B", 10, 4))
    assert plan.nodes[0].args["out_features"] == 10
    assert plan.nodes[1].input_shapes["x"] == ("B", 4, 10)
    assert plan.nodes[1].output_shapes["out"] == ("B", 10, 4)


def test_backward_inference_through_a_rank_four_reorder():
    plan = resolve("conv(kernel_size=3, padding=1)\npermute(2, 3, 1)", input_shape=("B", 3, 8, 8),
                   output_shape=("B", 8, 8, 5))
    assert plan.nodes[0].args["out_channels"] == 5
    assert plan.nodes[1].input_shapes["x"] == ("B", 5, 8, 8)


def test_dims_may_be_given_positionally_or_by_keyword():
    positional = resolve("permute(3, 1, 2)", input_shape=("B", 3, 5, 7), output_shape=("B", 7, 3, 5))
    keyword = resolve("permute(dims=(3, 1, 2))", input_shape=("B", 3, 5, 7), output_shape=("B", 7, 3, 5))
    assert positional.semantic_digest == keyword.semantic_digest
    assert positional.nodes[0].args["dims"] == (3, 1, 2)


@pytest.mark.parametrize("source,input_shape,output_shape,code", [
    ("permute()", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(0, 1)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(1, 0)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(1, 1)", ("B", 4, 6), ("B", 4, 6), "E_ARGUMENT"),
    ("permute(1, 3)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(2, 3)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(1, 2, 3, 4)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(2, 1)", ("B", 3, 5, 7), ("B", 3, 5, 7), "E_ARGUMENT"),
    ("permute(3, 1, 2)", ("B", 4, 6), ("B", 6, 4), "E_ARGUMENT"),
    ("permute(2, 1)", ("B", 4, 6), ("B", 4, 6), "E_CONSTRAINT"),
])
def test_invalid_axis_lists_and_contradictions_report_their_code(source, input_shape, output_shape, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=input_shape, output_shape=output_shape)
