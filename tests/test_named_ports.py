"""Named external inputs and outputs across both frontends and the backend."""

import copy
import json

import pytest
import torch
from torch import nn

import hndl
from hndl import HNDLError, ResolvedPlan, ops, resolve, resolve_callable
from hndl.torch import build, network, network_from_callable


DISCRIMINATOR = '''
flat = flatten(x)
joined = concat(flat, y, name="joined")
linear(joined, 32, name="body")
features = relu(name="features")
logits = linear(1, name="head")
'''

INPUTS = {"x": ("B", 1, 4, 4), "y": ("B", 10)}
OUTPUTS = {"logits": ("B", 1), "features": ("B", 32)}


def rehash(data):
    """Recompute both digests so schema checks, not checksums, are exercised."""
    from hndl.types import canonical, digest

    semantic = {key: value for key, value in data.items()
                if key not in {"frontend", "artifact_digest", "semantic_digest"}}
    semantic["nodes"] = [{key: value for key, value in node.items()
                          if key not in {"source", "provenance"}} for node in data["nodes"]]
    data["semantic_digest"] = digest(semantic)
    data.pop("artifact_digest", None)
    data["artifact_digest"] = digest(data)
    return canonical(data)


def discriminator(x, y):
    flat = ops.flatten(x)
    ops.concat(flat, y, name="joined")
    ops.linear(32, name="body")
    features = ops.relu(name="features")
    return {"logits": ops.linear(1, name="head"), "features": features}


def test_both_frontends_resolve_two_inputs_and_two_outputs():
    config_plan = resolve(DISCRIMINATOR, input_shape=INPUTS, output_shape=OUTPUTS)
    native_plan = resolve_callable(discriminator, input_shape=INPUTS, output_shape=OUTPUTS)
    assert config_plan.semantic_digest == native_plan.semantic_digest
    assert {name: entry["shape"] for name, entry in config_plan.inputs.items()} == INPUTS
    assert {name: entry["ref"] for name, entry in config_plan.outputs.items()} == {
        "logits": "node:head/out", "features": "node:features/out"}
    assert config_plan.nodes[0].inputs["x"] == "input:x"
    assert config_plan.nodes[1].inputs["x1"] == "input:y"
    # The leading declared input and output remain the legacy scalar fields.
    assert config_plan.input_shape == ("B", 1, 4, 4)
    assert config_plan.output_shape == ("B", 1) and config_plan.output_ref == "node:head/out"
    header = repr(config_plan).splitlines()[0]
    assert header == "Network: x=[B, 1, 4, 4], y=[B, 10] -> logits=[B, 1], features=[B, 32]  dtype=float32"


def test_multi_port_plan_round_trips_through_json():
    plan = resolve(DISCRIMINATOR, input_shape=INPUTS, output_shape=OUTPUTS)
    encoded = plan.to_json()
    data = json.loads(encoded)
    assert data["inputs"] == [{"name": "x", "shape": ["B", 1, 4, 4], "dtype": "float32"},
                              {"name": "y", "shape": ["B", 10], "dtype": "float32"}]
    assert data["outputs"] == [{"name": "logits", "ref": "node:head/out", "shape": ["B", 1]},
                               {"name": "features", "ref": "node:features/out", "shape": ["B", 32]}]
    restored = ResolvedPlan.from_json(encoded)
    assert restored.to_json() == encoded
    assert restored.semantic_digest == plan.semantic_digest
    assert tuple(restored.inputs) == ("x", "y") and tuple(restored.outputs) == ("logits", "features")


def test_single_input_and_output_plans_keep_the_original_encoding():
    # One input called x and one output called output is the legacy contract,
    # however it was spelled; only the selection rule differs.
    named = resolve("output = linear(64)", input_shape={"x": ("B", 128)},
                    output_shape={"output": ("B", 64)})
    plain = resolve("output = linear(64)", input_shape=("B", 128), output_shape=("B", 64))
    assert named.to_json() == plain.to_json()
    assert "inputs" not in json.loads(plain.to_json())
    assert named.named_inputs is None and named.named_outputs is None


def test_saved_multi_port_fields_are_validated():
    plan = resolve(DISCRIMINATOR, input_shape=INPUTS, output_shape=OUTPUTS)
    data = json.loads(plan.to_json())
    data["outputs"][0]["ref"] = "node:body/out"
    with pytest.raises(HNDLError, match="E_INTEGRITY"):
        ResolvedPlan.from_json(json.dumps(data))
    broken = json.loads(plan.to_json())
    broken["inputs"][0].pop("dtype")
    with pytest.raises(HNDLError, match="E_SCHEMA"):
        ResolvedPlan.from_json(rehash(broken))
    duplicated = json.loads(plan.to_json())
    duplicated["outputs"][1]["name"] = "logits"
    with pytest.raises(HNDLError, match="E_SCHEMA"):
        ResolvedPlan.from_json(rehash(duplicated))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_named_ports_build_and_run_with_real_tensors(device):
    model = network(DISCRIMINATOR, input_shape=INPUTS, output_shape=OUTPUTS,
                    device=device, initialization_seed=3)
    x = torch.randn(2, 1, 4, 4, device=device, requires_grad=True)
    y = torch.randn(2, 10, device=device, requires_grad=True)
    result = model(x, y)
    assert list(result) == ["logits", "features"]
    by_keyword = model(y=y, x=x)
    torch.testing.assert_close(result["logits"], by_keyword["logits"])
    expected = model["features"](model["body"](torch.cat((x.reshape(2, 16), y), dim=1)))
    torch.testing.assert_close(result["features"], expected)
    torch.testing.assert_close(result["logits"], model["head"](expected))
    (result["logits"].square().sum() + result["features"].square().sum()).backward()
    assert x.grad.isfinite().all() and y.grad.isfinite().all()
    # build() keeps the dictionary interface for every plan.
    graph = build(model.plan, device=device, initialization_seed=3)
    graph.load_state_dict(model.state_dict())
    torch.testing.assert_close(graph(x=x, y=y)["logits"], result["logits"])
    copied = copy.deepcopy(model)
    assert copied.plan is model.plan and copied._input_names == model._input_names
    torch.testing.assert_close(copied(x, y)["features"], result["features"])
    with pytest.raises(TypeError):
        len(model)


def test_named_inputs_with_one_output_still_return_a_tensor():
    model = network("concat(z, y)", input_shape={"z": ("B", 4), "y": ("B", 6)},
                    output_shape=("B", 10), device="cpu")
    result = model(torch.ones(2, 4), torch.zeros(2, 6))
    assert isinstance(result, torch.Tensor) and tuple(result.shape) == (2, 10)
    chain = network("linear(4)", input_shape={"z": ("B", 8)}, output_shape=("B", 4), device="cpu")
    assert len(chain) == 1 and chain[0] is chain["n0"]
    assert repr(chain).splitlines()[0] == "Network: z=[B, 8] -> [B, 4]  dtype=float32"


def test_per_input_dtypes_and_header():
    plan = resolve("embed = embedding(x, 6)\nconcat(embed, y, axis=2)",
                   input_shape={"x": ("B", 3), "y": ("B", 3, 2)},
                   output_shape=("B", 3, 8), input_dtype={"x": "int64"})
    assert plan.inputs["x"]["dtype"] == "int64" and plan.inputs["y"]["dtype"] == "float32"
    assert "input_dtype=x=int64" in repr(plan).splitlines()[0]
    model = build(plan, device="cpu")
    result = model(x=torch.randint(0, 6, (2, 3)), y=torch.randn(2, 3, 2))
    assert tuple(result["output"].shape) == (2, 3, 8)
    with pytest.raises(HNDLError, match="E_RUNTIME"):
        model(x=torch.randn(2, 3), y=torch.randn(2, 3, 2))
    with pytest.raises(HNDLError, match="E_DTYPE"):
        resolve("embed = embedding(x, 6)\nconcat(embed, y, axis=2)",
                input_shape={"x": ("B", 3), "y": ("B", 3, 2)}, output_shape=("B", 3, 8))


def test_callable_receives_named_inputs_as_keywords():
    seen = {}

    def author(*, z, y):
        seen.update(z=z.ref, y=y.ref)
        ops.concat(z, y)

    plan = resolve_callable(author, input_shape={"z": ("B", 4), "y": ("B", 2)},
                            output_shape=("B", 6))
    assert seen == {"z": "input:z", "y": "input:y"}
    assert plan.nodes[0].inputs == {"x0": "input:z", "x1": "input:y"}


def test_current_starts_at_the_first_declared_input():
    plan = resolve("h = linear(3)\nout = add(h, linear(y, 3))",
                   input_shape={"x": ("B", 5), "y": ("B", 7)}, output_shape=("B", 3))
    assert plan.nodes[0].inputs["x"] == "input:x"
    assert plan.nodes[1].inputs["x"] == "input:y"


@pytest.mark.parametrize("kwargs,code,message", [
    (dict(input_shape={"x": ("B", 4), "y": ("B", 4)}, output_shape=("B", 3)),
     "E_BINDING", "never used"),
    (dict(input_shape={"x": ("B", 4), "y": (2, 4)}, output_shape=("B", 3)),
     "E_CONSTRAINT", "same batch dimension"),
    (dict(input_shape=("B", 4), output_shape={"relu": ("B", 3)}),
     "E_NAME", "collide with operator aliases"),
    (dict(input_shape={"linear": ("B", 4)}, output_shape=("B", 3)),
     "E_NAME", "collide with operator aliases"),
    (dict(input_shape={"Bad": ("B", 4)}, output_shape=("B", 3)), "E_SCHEMA", "must match"),
    (dict(input_shape={}, output_shape=("B", 3)), "E_SCHEMA", "must name between"),
    (dict(input_shape=("B", 4), output_shape=("B", 3), input_dtype={"z": "int64"}),
     "E_SCHEMA", "undeclared inputs"),
    (dict(input_shape=("B", 4), output_shape={"first": ("B", 3), "second": ("B", 3)}),
     "E_OUTPUT", "never selected"),
])
def test_named_port_declaration_errors(kwargs, code, message):
    with pytest.raises(HNDLError, match=code) as failure:
        resolve("linear(3)", **kwargs)
    assert message in failure.value.message


def test_named_output_must_be_one_symbol_from_this_capture():
    with pytest.raises(HNDLError, match="E_OUTPUT"):
        resolve("parts = split(2)\nfirst = parts\nsecond = parts",
                input_shape=("B", 4), output_shape={"first": ("B", 2), "second": ("B", 2)})
    with pytest.raises(HNDLError, match="E_OUTPUT"):
        resolve_callable(lambda x: ops.linear(3), input_shape=("B", 4),
                         output_shape={"only": ("B", 3)})
    with pytest.raises(HNDLError, match="E_OUTPUT"):
        resolve_callable(lambda x: {"only": ops.linear(3), "extra": x}, input_shape=("B", 4),
                         output_shape={"only": ("B", 3)})


def test_runtime_binding_errors_name_the_declared_inputs():
    model = network("concat(x, y)", input_shape={"x": ("B", 4), "y": ("B", 6)},
                    output_shape=("B", 10), device="cpu")
    for call in (lambda: model(torch.randn(2, 4)),
                 lambda: model(x=torch.randn(2, 4)),
                 lambda: model(torch.randn(2, 4), torch.randn(2, 6), torch.randn(2, 6)),
                 lambda: model(x=torch.randn(2, 4), z=torch.randn(2, 6))):
        with pytest.raises(HNDLError, match="E_BINDING"):
            call()
    with pytest.raises(HNDLError, match="E_BINDING"):
        model(torch.randn(2, 4), x=torch.randn(2, 4))
    graph = build(model.plan, device="cpu")
    with pytest.raises(HNDLError, match="E_BINDING"):
        graph(x=torch.randn(2, 4))


def test_named_outputs_may_share_one_reference_and_reach_an_input():
    plan = resolve("h = relu(x)\nfirst = h\nsecond = h\npassthrough = y",
                   input_shape={"x": ("B", 4), "y": ("B", 2)},
                   output_shape={"first": ("B", 4), "second": ("B", 4), "passthrough": ("B", 2)})
    assert [entry["ref"] for entry in plan.outputs.values()] == [
        "node:n0/out", "node:n0/out", "input:y"]
    model = build(plan, device="cpu")
    x, y = torch.randn(2, 4), torch.randn(2, 2)
    result = model(x=x, y=y)
    assert result["first"] is result["second"] and result["passthrough"] is y


def test_custom_operator_and_parameter_counts_with_named_ports():
    registry = hndl.Registry.builtins()

    @registry.operator("keep", identity="example.keep", summary="Identity.",
                       shape="x[B, ...] -> out[B, ...]")
    class Keep(nn.Identity):
        pass

    model = network_from_callable(
        lambda *, a, b: {"total": ops.add(registry.ops.keep(a), b)},
        input_shape={"a": ("B", 3), "b": ("B", 3)}, output_shape={"total": ("B", 3)},
        device="cpu", registry=registry)
    from hndl.torch import parameter_counts
    assert parameter_counts(model.plan, registry=registry) == {"n0": 0, "n1": 0}
    result = model(torch.ones(2, 3), torch.ones(2, 3))
    torch.testing.assert_close(result, torch.full((2, 3), 2.0))
