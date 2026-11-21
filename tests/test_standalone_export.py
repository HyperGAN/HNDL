import pytest
import torch

from hndl import HNDLError, Registry, resolve
from hndl.torch import build
from tests._network_examples import load, names, random_inputs


@pytest.mark.parametrize("name", names())
def test_standalone_shipped_recipe(name):
    source, _, kwargs = load(name)
    plan = resolve(source, **kwargs)
    namespace = {}
    exec(plan.to_python_source(), namespace)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        reference = build(plan, device="cpu").eval()
        torch.manual_seed(7)
        generated = namespace["Model"]().eval()
    torch.testing.assert_close(generated.state_dict(), reference.state_dict(), rtol=0, atol=0)
    assert [p.requires_grad for p in generated.parameters()] == [p.requires_grad for p in reference.parameters()]
    generated.load_state_dict(reference.state_dict())
    inputs = random_inputs(plan, batch_size=1)
    with torch.no_grad():
        torch.testing.assert_close(generated(**inputs), reference(*inputs.values()))


def test_standalone_settings_apply_before_loading_weights():
    plan = resolve('linear(4, init={"weight": uniform(a=0.5,b=0.5), "bias": 2}, trainable={"weight": False})',
                   input_shape=("B", 4), output_shape=("B", 4), dtype="bfloat16")
    namespace = {}
    exec(plan.to_python_source(), namespace)
    layer = namespace["Model"]().nodes["n_n0"]
    assert layer.weight.dtype == torch.bfloat16 and not layer.weight.requires_grad and layer.bias.requires_grad
    torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, .5))
    torch.testing.assert_close(layer.bias, torch.full_like(layer.bias, 2))


def test_standalone_parameter_ties_inside_expert_lists():
    plan = resolve('moe(experts=4, hidden=8, name="first"); '
                   'moe(experts=4, hidden=8, tie={"experts.0.up.weight": "first.experts.1.up.weight"})',
                   input_shape=("B", 8), output_shape=("B", 8))
    namespace = {}
    exec(plan.to_python_source(), namespace)
    generated = namespace["Model"]()
    first, second = generated.nodes.values()
    assert first.experts[1].up.weight is second.experts[0].up.weight
    reference = build(plan, device="cpu")
    generated.load_state_dict(reference.state_dict())
    x = torch.randn(4, 8)
    torch.testing.assert_close(generated(x), reference(x))


def test_standalone_reports_custom_runtime_dependency():
    registry = Registry.builtins()
    gain = object()

    @registry.operator("custom_gain", shape="x[B,D] -> out[B,D]", summary="Custom gain.")
    class CustomGain(torch.nn.Module):
        def forward(self, x):
            return x * gain

    plan = resolve("custom_gain()", input_shape=("B", 4), output_shape=("B", 4), registry=registry)
    with pytest.raises(HNDLError, match="Cannot emit standalone custom_gain") as caught:
        plan.to_python_source()
    assert caught.value.node == plan.nodes[0].id


def test_standalone_custom_mapping_and_graph_input_outputs():
    registry = Registry.builtins()

    @registry.operator("custom_pair", shape="x[B,D] -> positive[B,D], negative[B,D]", summary="Signed pair.")
    class CustomPair(torch.nn.Module):
        def forward(self, x):
            return {"negative": -x, "positive": x}

    shape = ("B", 4)
    plan = resolve("p, n = custom_pair(x)\nfirst = p\nsecond = n\noriginal = x",
                   input_shape={"x": shape}, output_shape=dict.fromkeys(("first", "second", "original"), shape),
                   registry=registry)
    namespace = {}
    exec(plan.to_python_source(), namespace)
    x = torch.randn(2, 4)
    torch.testing.assert_close(namespace["Model"]()(x), {"first": x, "second": -x, "original": x})
