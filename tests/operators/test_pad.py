"""Numerics, bidirectional inference, and error codes for the ``pad`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.operators.pad import Pad
from hndl.torch import DTYPES, build

from .conftest import DEVICES


CASES = [
    ((3, 7), (2, 1), "constant", 0.0),
    ((3, 7), (0, 4), "constant", -1.5),
    ((2, 3, 8), (2, 3), "constant", 0.25),
    ((2, 3, 8), (2, 2), "reflect", 0.0),
    ((2, 3, 8), (3, 0), "replicate", 0.0),
    ((2, 3, 5, 6), (1, 1, 2, 2), "constant", 0.0),
    ((2, 3, 5, 6), (2, 1, 1, 3), "reflect", 0.0),
    ((2, 3, 5, 6), (0, 2, 3, 0), "replicate", 0.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,padding,mode,value", CASES)
def test_matches_functional_pad_and_its_gradient(device, shape, padding, mode, value):
    module = Pad(padding=padding, value=value, mode=mode).to(device)
    x = torch.randn(*shape, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()

    actual = module(x)
    expected = (F.pad(mirror, padding, mode="constant", value=value) if mode == "constant"
                else F.pad(mirror, padding, mode=mode))
    torch.testing.assert_close(actual, expected)

    totals = [0] * len(shape)
    for index in range(len(padding) // 2):
        totals[len(shape) - 1 - index] = padding[2 * index] + padding[2 * index + 1]
    assert tuple(actual.shape) == tuple(s + t for s, t in zip(shape, totals))

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_constant_padding_copies_the_interior_and_fills_the_border(device):
    plan = resolve('pad(1, 2, value=7.0)', input_shape=("B", 2, 3, 4), output_shape=("B", 2, 3, 7))
    model = build(plan, device=device)
    x = torch.randn(3, 2, 3, 4, device=device)
    output = model(x=x)["output"]
    torch.testing.assert_close(output[..., 1:5], x)
    assert torch.equal(output[..., :1], torch.full_like(output[..., :1], 7.0))
    assert torch.equal(output[..., 5:], torch.full_like(output[..., 5:], 7.0))


@pytest.mark.parametrize("device", DEVICES)
def test_reflect_mirrors_without_repeating_the_edge(device):
    module = Pad(padding=(2, 2), value=0.0, mode="reflect").to(device)
    x = torch.arange(5.0, device=device).reshape(1, 1, 5)
    expected = torch.tensor([[[2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]]], device=device)
    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_replicate_repeats_the_edge(device):
    module = Pad(padding=(2, 1), value=0.0, mode="replicate").to(device)
    x = torch.arange(4.0, device=device).reshape(1, 1, 4)
    expected = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0]]], device=device)
    torch.testing.assert_close(module(x), expected)


def test_output_extent_is_inferred_forward_and_input_extent_backward():
    forward = resolve("pad(1, 1, 2, 2)", input_shape=("B", 3, 8, 8), output_shape=("B", 3, 12, 10))
    assert forward.nodes[0].output_shapes["out"] == ("B", 3, 12, 10)

    backward = resolve("linear()\npad(3, 4)", input_shape=("B", 16), output_shape=("B", 20))
    assert backward.nodes[0].args["out_features"] == 13
    assert backward.nodes[1].input_shapes["x"] == ("B", 13)

    through = resolve("pad(1, 1)\nlinear()", input_shape=("B", 5, 6), output_shape=("B", 5, 3))
    assert through.nodes[0].output_shapes["out"] == ("B", 5, 8)
    assert through.nodes[1].args["in_features"] == 8

    sandwich = resolve("pad(0, 2, 0, 2)\nconv(4, kernel_size=3)", input_shape=("B", 3, 6, 6),
                       output_shape=("B", 4, 6, 6))
    assert sandwich.nodes[0].output_shapes["out"] == ("B", 3, 8, 8)


def test_padding_is_unbounded_by_the_element_count_only_through_the_declared_shapes():
    plan = resolve("pad(2, 2)", input_shape=("B", 4), output_shape=("B", 8))
    assert plan.nodes[0].args["padding"] == (2, 2)
    model = build(plan, device="cpu")
    x = torch.randn(2, 4)
    torch.testing.assert_close(model(x=x)["output"], F.pad(x, (2, 2), mode="constant", value=0.0))


@pytest.mark.parametrize("source,shapes,code,message", [
    ("pad(1, 1, 1)", (("B", 4), ("B", 7)), "E_ARGUMENT", "even number of values"),
    ("pad()", (("B", 4), ("B", 4)), "E_ARGUMENT", "even number of values"),
    ("pad(-1, 2)", (("B", 4), ("B", 5)), "E_ARGUMENT", "must be >= 0"),
    ("pad(1, -2)", (("B", 4), ("B", 3)), "E_ARGUMENT", "must be >= 0"),
    ("pad(1, 1, 1, 1)", (("B", 4), ("B", 6)), "E_ARGUMENT", "only 1 non-batch axes"),
    ("pad(1, 1, 1, 1, 1, 1)", (("B", 3, 4), ("B", 5, 6)), "E_ARGUMENT", "only 2 non-batch axes"),
    ('pad(1, 1, mode="reflect")', (("B", 6), ("B", 8)), "E_ARGUMENT", "pads the spatial axes only"),
    ('pad(1, 1, mode="reflect")', (("B", 2, 4, 4), ("B", 2, 4, 6)), "E_ARGUMENT", "pads the spatial axes only"),
    ('pad(1, 1, 1, 1, mode="replicate")', (("B", 2, 6), ("B", 4, 8)), "E_ARGUMENT", "pads the spatial axes only"),
    ('pad(1, 1, value=2.0, mode="replicate")', (("B", 2, 4), ("B", 2, 6)), "E_ARGUMENT", "mode='constant' only"),
    ('pad(3, 0, mode="reflect")', (("B", 2, 3), ("B", 2, 6)), "E_CONSTRAINT", "must be smaller than the input extent"),
    ('pad(0, 4, mode="reflect")', (("B", 2, 4), ("B", 2, 8)), "E_CONSTRAINT", "must be smaller than the input extent"),
    ('pad(1, 1, mode="banana")', (("B", 2, 4), ("B", 2, 6)), "E_ARGUMENT", "must be one of"),
])
def test_invalid_declarations_report_their_error_code(source, shapes, code, message):
    with pytest.raises(HNDLError, match=code) as failure:
        resolve(source, input_shape=shapes[0], output_shape=shapes[1])
    assert message in failure.value.message


def test_output_contract_smaller_than_the_padding_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT") as failure:
        resolve("linear()\npad(4, 4)", input_shape=("B", 8), output_shape=("B", 6))
    assert "padded positions" in failure.value.message


def test_reflect_padding_equal_to_the_extent_is_rejected_where_torch_would_fail():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve('pad(4, 4, mode="reflect")', input_shape=("B", 2, 4), output_shape=("B", 2, 12))
    ok = resolve('pad(3, 3, mode="reflect")', input_shape=("B", 2, 4), output_shape=("B", 2, 10))
    model = build(ok, device="cpu")
    x = torch.randn(2, 2, 4)
    torch.testing.assert_close(model(x=x)["output"], F.pad(x, (3, 3), mode="reflect"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
def test_reduced_precision_padding_stays_in_the_input_dtype(dtype, mode):
    source = f'pad(1, 1, 1, 1, mode="{mode}")'
    plan = resolve(source, input_shape=("B", 2, 5, 5), output_shape=("B", 2, 7, 7), dtype=dtype)
    model = build(plan, device="cuda:0")
    torch_dtype = DTYPES[dtype]
    x = torch.randn(3, 2, 5, 5, device="cuda:0", dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (3, 2, 7, 7)
    expected = (F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0) if mode == "constant"
                else F.pad(x, (1, 1, 1, 1), mode=mode))
    torch.testing.assert_close(output, expected)
