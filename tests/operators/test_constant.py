"""Numerics, inference, and error codes for ``constant``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build

SHAPES = [
    pytest.param((6,), id="rank2"),
    pytest.param((5, 8), id="rank3"),
    pytest.param((4, 6, 6), id="rank4"),
]


def constant_plan(shape, value=None, **kwargs):
    arguments = ", ".join(str(extent) for extent in shape)
    if value is not None:
        arguments += f", value={value}"
    return resolve(f"constant({arguments})", input_shape=("B", 4),
                   output_shape=("B", *shape), **kwargs)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", [0.0, 1.0, -2.5])
def test_every_element_carries_the_value(shape, value, device):
    model = build(constant_plan(shape, value), device=device)
    x = torch.randn(3, 4, device=device, requires_grad=True)
    output = model(x=x)["output"]
    torch.testing.assert_close(output, torch.full((3, *shape), value, device=device))
    assert output.dtype == x.dtype and output.device == x.device
    assert not output.requires_grad and output.grad_fn is None


def test_the_default_value_is_zero(device):
    plan = resolve("constant(8)", input_shape=("B", 4), output_shape=("B", 8))
    assert plan.nodes[0].args["value"] == 0.0
    model = build(plan, device=device)
    output = model(x=torch.randn(2, 4, device=device))["output"]
    assert torch.equal(output, torch.zeros(2, 8, device=device))


@pytest.mark.parametrize("batch", [1, 2, 7])
def test_the_batch_follows_the_runtime_batch(batch, device):
    model = build(constant_plan((3, 5)), device=device)
    output = model(x=torch.randn(batch, 4, device=device))["output"]
    assert tuple(output.shape) == (batch, 3, 5)


def test_the_node_is_stateless_and_reports_its_arguments(device):
    model = build(constant_plan((6,), 1.5), device=device)
    node = model["n0"]
    assert list(node.parameters()) == [] and list(node.buffers()) == []
    assert node.state_dict() == {}
    assert "shape=(6,)" in repr(node) and "value=1.5" in repr(node)
    assert model.build_receipt["state_bytes"] == 0


def test_train_and_eval_agree_and_repeat(device):
    model = build(constant_plan((4, 2), 0.25), device=device)
    x = torch.randn(2, 4, device=device)
    model.train()
    trained = model(x=x)["output"]
    model.eval()
    torch.testing.assert_close(model(x=x)["output"], trained)
    torch.testing.assert_close(model(x=x)["output"], trained)


def test_no_gradient_reaches_the_input(device):
    source = "h = linear(8)\nz = constant(8, value=2.0)\nmul(h, z)"
    plan = resolve(source, input_shape=("B", 4), output_shape=("B", 8))
    model = build(plan, device=device, initialization_seed=5)
    x = torch.randn(2, 4, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    output = model(x=x)["output"]
    torch.testing.assert_close(output, 2.0 * model["n0"](mirror))
    output.square().mean().backward()
    (2.0 * model["n0"](mirror)).square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_the_shape_is_inferred_from_the_output_contract():
    plan = resolve("constant()", input_shape=("B", 4), output_shape=("B", 3, 6, 6))
    assert plan.nodes[0].args["shape"] == (3, 6, 6)
    assert plan.nodes[0].output_shapes["out"] == ("B", 3, 6, 6)
    assert plan.nodes[0].provenance["shape"] == "operator default"


def test_the_shape_is_inferred_from_a_consuming_node():
    plan = resolve("h = linear(8)\nz = constant()\nadd(h, z)",
                   input_shape=("B", 4), output_shape=("B", 8))
    assert plan.nodes[1].args["shape"] == (8,)


def test_an_explicit_shape_fixes_the_output():
    plan = resolve("constant(512, 4, 4)\nconv(3, kernel_size=1)",
                   input_shape=("B", 16), output_shape=("B", 3, 4, 4))
    assert plan.nodes[0].output_shapes["out"] == ("B", 512, 4, 4)
    assert plan.nodes[0].input_shapes["x"] == ("B", 16)


def test_an_unconstrained_shape_is_reported_as_ambiguous():
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("z = constant()\nlinear(z, 4)", input_shape=("B", 4), output_shape=("B", 4))


@pytest.mark.parametrize("source,code", [
    ("constant(0)", "E_ARGUMENT"),
    ("constant(-4)", "E_ARGUMENT"),
    ("constant(8, 0)", "E_ARGUMENT"),
    ("constant(2, 2, 2, 2)", "E_ARGUMENT"),
    ("constant(4.5)", "E_ARGUMENT"),
    ("constant('8')", "E_ARGUMENT"),
    ("constant(8, value='high')", "E_ARGUMENT"),
    ("constant(8, value=True)", "E_ARGUMENT"),
])
def test_invalid_arguments_report_their_code(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 4), output_shape=("B", 8))


@pytest.mark.parametrize("source,shape", [
    ("h = linear(8)\nz = constant(6)\nadd(h, z)", ("B", 8)),
    ("constant(4, 4, 4)", ("B", 8)),
])
def test_a_shape_that_contradicts_the_contract_fails(source, shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve(source, input_shape=("B", 4), output_shape=shape)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_reduced_precision_plans_emit_the_compute_dtype(dtype, device):
    plan = resolve("constant(6, value=0.5)", input_shape=("B", 4),
                   output_shape=("B", 6), dtype=dtype)
    model = build(plan, device=device)
    torch_dtype = DTYPES[dtype]
    x = torch.randn(2, 4, device=device, dtype=torch_dtype)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype
    torch.testing.assert_close(output, torch.full((2, 6), 0.5, device=device, dtype=torch_dtype))


def test_an_integer_graph_input_still_yields_the_compute_dtype(device):
    plan = resolve("z = constant(4, value=1.0)\nlinear(z, 2)", input_shape=("B", 3),
                   output_shape=("B", 2), input_dtype="int64")
    model = build(plan, device=device, initialization_seed=1)
    ids = torch.randint(0, 8, (2, 3), device=device)
    constant = model["n0"](ids)
    assert constant.dtype == torch.float32
    assert torch.equal(constant, torch.ones(2, 4, device=device))
    assert model(x=ids)["output"].shape == (2, 2)


def test_the_constant_moves_with_the_model():
    if not torch.cuda.is_available():
        pytest.skip("device movement needs a second device")
    model = build(constant_plan((6,), 3.0), device="cpu").to("cuda:0")
    output = model(x=torch.randn(2, 4, device="cuda:0"))["output"]
    assert output.device.type == "cuda" and torch.equal(
        output, torch.full((2, 6), 3.0, device="cuda:0"))
