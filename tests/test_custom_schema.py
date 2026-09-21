from dataclasses import FrozenInstanceError
import json

import pytest

from hndl import Argument, Dim, HNDLError, Registry, ResolvedPlan, ShapeRule, preserves_shape, resolve
from hndl.registry import normalize_arguments
from hndl.resolver import resolve_graph
from hndl.types import Graph, Node


def adain_registry():
    registry = Registry.builtins()
    registry.register(
        "adaptive_norm", identity="example.adaptive_norm", version=1, max_state_bytes=0,
        shape=ShapeRule(
            inputs={"x": ("B", "C", "H", "W"), "params": ("B", Dim("C", scale=2))},
            outputs={"out": ("B", "C", "H", "W")},
        ),
        arguments={"eps": Argument(float, default=1e-5, minimum=0, exclusive_minimum=True)},
    )
    return registry


def test_adain_rule_infers_style_width_backward_and_roundtrips():
    registry = adain_registry()
    plan = resolve(
        'w = linear(256)\nlinear(w)\nfeatures = reshape(64,4,4)\nstyle = linear(w)\nadaptive_norm(features, style)',
        input_shape=("B", 128), output_shape=("B", 64, 4, 4), registry=registry,
    )
    assert plan.nodes[1].args["out_features"] == 1024
    assert plan.nodes[3].args["out_features"] == 128
    assert plan.nodes[-1].args == {"eps": 1e-5}
    assert plan.nodes[-1].state_bytes == 0
    assert ResolvedPlan.from_json(plan.to_json(), registry=registry).semantic_digest == plan.semantic_digest
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        ResolvedPlan.from_json(plan.to_json())


def test_shared_scale_infers_unknown_feature_channels_from_style_port():
    registry = adain_registry()
    nodes = (
        Node("style", "linear@1", {"out_features": 128}, {"x": "input:x"}),
        Node("project", "linear@1", {}, {"x": "input:x"}),
        Node("seed", "reshape@1", {}, {"x": "node:project/out"}),
        Node("norm", "example.adaptive_norm@1", {}, {"x": "node:seed/out", "params": "node:style/out"}),
        Node("flatten", "flatten@1", {}, {"x": "node:norm/out"}),
    )
    # Fix height in a separate literal rule, leaving the other axis to the
    # existing product relation after C is determined by the style width.
    registry.register("height", identity="example.height", version=1, max_state_bytes=0,
                      shape=ShapeRule(inputs={"x": ("B", "C", 4, "W")}, outputs={"out": ("B", "C", 4, "W")}))
    nodes = (*nodes[:4], Node("height", "example.height@1", {}, {"x": "node:norm/out"}),
             Node("flatten", "flatten@1", {}, {"x": "node:height/out"}))
    plan = resolve_graph(Graph(nodes, ("B", 128), ("B", 1024), "node:flatten/out"), registry)
    assert plan.nodes[2].args["shape"] == (64, 4, 4)
    assert plan.nodes[1].args["out_features"] == 1024


def test_scaled_contradictions_have_node_and_config_source():
    with pytest.raises(HNDLError, match="divisible") as caught:
        resolve('z1,z2=split(63)\nlinear(z1)\nf=reshape(32,4,4)\nadaptive_norm(f,z2)',
                input_shape=("B", 128), output_shape=("B", 32, 4, 4), registry=adain_registry())
    assert caught.value.node == "n3"
    assert caught.value.line == 4
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve('z1,z2=split(64)\nlinear(z1)\nf=reshape(16,4,4)\nadaptive_norm(f,z2)',
                input_shape=("B", 128), output_shape=("B", 16, 4, 4), registry=adain_registry())


def test_scaled_rules_work_forward_backward_and_symbols_are_per_node():
    registry = Registry.builtins()
    registry.register("double", identity="example.double", version=1, max_state_bytes=0,
                      shape=ShapeRule(inputs={"data": ("B", "F")}, outputs={"value": ("B", Dim("F", 2))}))
    plan = resolve('double()\ndouble()', input_shape=("B", 3), output_shape=("B", 12), registry=registry)
    assert plan.nodes[0].output_shapes["value"] == ("B", 6)
    assert plan.nodes[1].output_shapes["value"] == ("B", 12)
    reverse = resolve('linear()\ndouble()', input_shape=("B", 3), output_shape=("B", 10), registry=registry)
    assert reverse.nodes[0].args["out_features"] == 5
    with pytest.raises(HNDLError, match="divisible"):
        resolve('linear()\ndouble()', input_shape=("B", 3), output_shape=("B", 9), registry=registry)


def test_multioutput_rule_and_literal_axis_constraints():
    registry = Registry.builtins()
    registry.register("pair", identity="example.pair", version=1, max_state_bytes=12,
                      shape=ShapeRule(inputs={"data": ("B", "F")},
                                      outputs={"wide": ("B", Dim("F", 2)), "fixed": ("B", 3)}))
    plan = resolve('wide,fixed=pair()\nout=wide', input_shape=("B", 5), output_shape=("B", 10), registry=registry)
    assert plan.nodes[0].outputs == ("wide", "fixed")
    assert plan.nodes[0].output_shapes["fixed"] == ("B", 3)
    assert plan.state_bytes == 12
    with pytest.raises(HNDLError, match="max_state_bytes"):
        resolve('wide,fixed=pair()\nout=wide', input_shape=("B", 5), output_shape=("B", 10),
                registry=registry, limits={"max_state_bytes": 8})


def test_named_preserves_shape_and_custom_builtin_like_argument_names():
    registry = Registry.builtins()
    registry.register("custom", identity="example.custom", version=1, max_state_bytes=8,
                      input_ports=("features",), output_ports=("value",), shape=preserves_shape,
                      arguments={"dim": Argument(str, default="channels"),
                                 "bias": Argument(int, default=-2),
                                 "kernel_size": Argument(bool, default=False),
                                 "eps": Argument(float, default=-1.0, maximum=0)})
    plan = resolve('linear()\ncustom()', input_shape=("B", 4), output_shape=("B", 8), registry=registry)
    assert plan.nodes[0].args["out_features"] == 8
    assert plan.nodes[1].args == {"dim": "channels", "bias": -2, "kernel_size": False, "eps": -1.0}
    assert plan.nodes[1].input_shapes == {"features": ("B", 8)}
    assert plan.nodes[1].state_bytes == 8


def test_schema_copies_and_freezes_patterns_arguments_and_defaults():
    inputs = {"x": ["B", "C"]}
    outputs = {"out": ["B", Dim("C", 2)]}
    rule = ShapeRule(inputs=inputs, outputs=outputs)
    args = {"gain": Argument(float, default=2)}
    registry = Registry.builtins()
    entry = registry.register("scale", identity="example.scale", version=1, shape=rule,
                              arguments=args, max_state_bytes=0)
    inputs["x"][1] = 999
    outputs.clear()
    args.clear()
    assert rule.inputs["x"] == (Dim("B"), Dim("C"))
    assert entry.defaults == {"gain": 2.0}
    with pytest.raises(TypeError):
        rule.inputs["new"] = ("B", 4)
    with pytest.raises(TypeError):
        entry.arguments["gain"] = Argument(float)
    with pytest.raises(FrozenInstanceError):
        rule.outputs["out"][1].scale = 3


@pytest.mark.parametrize("factory", [
    lambda: Argument(list), lambda: Argument(int, default=True),
    lambda: Argument(float, default=float("nan")), lambda: Argument(float, default=float("inf")),
    lambda: Argument(int, required=True, default=1), lambda: Argument(int, required=False),
    lambda: Argument(float, minimum=3, maximum=2),
    lambda: Argument(int, minimum=0.1, maximum=0.9),
    lambda: Argument(float, minimum=1, maximum=1, exclusive_minimum=True),
    lambda: Argument(float, exclusive_minimum=True), lambda: Argument(bool, minimum=0),
    lambda: Argument(str, default="x" * 16_385), lambda: Argument(str, default="\ud800"),
    lambda: Argument(int, default=2**63),
])
def test_invalid_scalar_schemas_reject_registration(factory):
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        factory()


def test_required_scalar_types_bounds_and_default_expansion():
    registry = Registry.builtins()
    entry = registry.register("custom", identity="example.custom", version=1, max_state_bytes=0,
                              shape=preserves_shape, arguments={
                                  "count": Argument(int, minimum=1),
                                  "gain": Argument(float, default=1, minimum=0, exclusive_minimum=True, maximum=2),
                                  "enabled": Argument(bool, default=True),
                                  "label": Argument(str, default="a # literal"),
                              })
    assert entry.required == ("count",)
    assert normalize_arguments(entry, (1,), {}) == {"count": 1, "gain": 1.0, "enabled": True, "label": "a # literal"}
    for args in ({}, {"count": True}, {"count": 0}, {"count": 1, "gain": 0},
                 {"count": 1, "gain": float("nan")}, {"count": 1, "gain": 3},
                 {"count": 1, "enabled": 1}, {"count": 1, "extra": 1}):
        with pytest.raises(HNDLError, match="E_ARGUMENT"):
            normalize_arguments(entry, (), args)


@pytest.mark.parametrize("factory", [
    lambda: Dim("C", 0), lambda: Dim("C", True), lambda: Dim("C", 2**31), lambda: Dim("B", 2),
    lambda: ShapeRule(inputs={"x": ("C", "B")}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={"x": (1, "C")}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={"x": ("B", True)}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={"x": ("B", 0)}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={"x": ("B", "C", "W")}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={"name": ("B", "C")}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={f"p{i}": ("B", "C") for i in range(33)}, outputs={"out": ("B", "C")}),
    lambda: ShapeRule(inputs={f"p{i}": ("B", f"C{i}", f"H{i}", f"W{i}") for i in range(22)}, outputs={"out": ("B", "C")}),
])
def test_invalid_or_excessive_shape_patterns_fail(factory):
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        factory()


def test_port_schema_conflicts_callbacks_and_underscored_aliases_fail():
    def register(**kwargs):
        return Registry.builtins().register("custom", identity="example.custom", version=1,
                                           max_state_bytes=0, **kwargs)
    for field in ("x", "out", "name", "policy"):
        with pytest.raises(HNDLError, match="conflicts"):
            register(shape=preserves_shape, arguments={field: Argument(int)})
    with pytest.raises(HNDLError, match="callbacks"):
        register(shape=lambda x: x)
    with pytest.raises(HNDLError, match="exactly one"):
        register(shape=preserves_shape, input_ports=("a", "b"))
    with pytest.raises(HNDLError, match="exactly match"):
        register(shape=ShapeRule(inputs={"a": ("B", "F"), "b": ("B", "F")}, outputs={"out": ("B", "F")}),
                 input_ports=("b", "a"))
    with pytest.raises(HNDLError, match="64 fields"):
        register(shape=preserves_shape, arguments={f"a{i}": Argument(int) for i in range(65)})
    for alias in ("_custom", "_registry"):
        with pytest.raises(HNDLError, match="E_REGISTRY"):
            Registry.builtins().register(alias, identity="example.custom", version=1, max_state_bytes=0, shape=preserves_shape)


def test_custom_pattern_resource_limits_apply_before_build():
    registry = Registry.builtins()
    registry.register("huge", identity="example.huge", version=1, max_state_bytes=0,
                      shape=ShapeRule(inputs={"x": ("B", "F")}, outputs={"out": ("B", Dim("F", 2**31 - 1))}))
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        resolve('huge()\nlinear()', input_shape=("B", 2), output_shape=("B", 1), registry=registry)


def test_restoration_rechecks_rule_incompatible_under_same_identity():
    registry = Registry.builtins()
    registry.register("twice", identity="example.twice", version=1, max_state_bytes=0,
                      shape=ShapeRule(inputs={"x": ("B", "F")}, outputs={"out": ("B", Dim("F", 2))}))
    plan = resolve('twice()', input_shape=("B", 4), output_shape=("B", 8), registry=registry)
    changed = Registry.builtins()
    changed.register("twice", identity="example.twice", version=1, max_state_bytes=0,
                     shape=ShapeRule(inputs={"x": ("B", "F")}, outputs={"out": ("B", Dim("F", 3))}))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        ResolvedPlan.from_json(plan.to_json(), registry=changed)
    assert json.loads(plan.to_json())["nodes"][0]["op"] == "example.twice@1"

