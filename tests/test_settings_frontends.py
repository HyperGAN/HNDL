"""Persist construction metadata consistently without executing a backend."""

import math
import struct

import pytest
import torch

from torch import nn

from hndl import Arg, HNDLError, Registry, ResolvedPlan, ops
from hndl import resolve, resolve_callable
from hndl import kaiming_normal, normal, orthogonal, truncated_normal, uniform, xavier_uniform
from hndl.capture import capture_callable
from hndl.config import _parse, capture_config
from hndl.initializers import kaiming_uniform, xavier_normal


SHAPES = {"input_shape": ("B", 8), "output_shape": ("B", 2)}


def test_config_native_settings_roundtrip_without_recapturing_or_torch():
    source = '''
    linear(4, init={"weight": 0, "bias": 1}, trainable=False, name="frozen")
    relu()
    linear(init={"bias": -0.25}, trainable={"weight": False}, name="head")
    '''
    calls = []

    def author(x):
        calls.append(1)
        ops.linear(4, init={"weight": 0, "bias": 1}, trainable=False, name="frozen")
        ops.relu()
        ops.linear(init={"bias": -0.25}, trainable={"weight": False}, name="head")

    config_plan = resolve(source, **SHAPES)
    native_plan = resolve_callable(author, **SHAPES)
    assert calls == [1]
    assert native_plan.semantic_digest == config_plan.semantic_digest
    restored = ResolvedPlan.from_json(native_plan.to_json())
    assert calls == [1]
    assert restored.semantic_digest == native_plan.semantic_digest
    frozen = restored.nodes[0]
    assert dict(frozen.initialization["overrides"]) == {"weight": 0.0, "bias": 1.0}
    assert frozen.initialization["kind"] == "torch_default@1"
    assert dict(frozen.trainability) == {"default": False, "overrides": {}}
    head = restored.nodes[-1]
    assert dict(head.initialization["overrides"]) == {"bias": -0.25}
    assert head.trainability["default"] is None
    assert dict(head.trainability["overrides"]) == {"weight": False}
    for node in restored.nodes:
        assert "init" not in node.args and "trainable" not in node.args
        assert "init" not in node.source["argument_origins"]
        assert "trainable" not in node.source["argument_origins"]


def test_empty_maps_are_semantically_default_and_preserve_constructor_masks():
    kwargs = {"input_shape": ("B", 8), "output_shape": ("B", 4)}
    default = resolve("linear(4)", **kwargs)
    empty = resolve("linear(4, init={}, trainable={})", **kwargs)
    native = resolve_callable(lambda x: ops.linear(4, init={}, trainable={}), **kwargs)
    assert default.semantic_digest == empty.semantic_digest == native.semantic_digest
    assert empty.nodes[0].trainability["default"] is None
    assert dict(empty.nodes[0].trainability["overrides"]) == {}
    assert dict(empty.nodes[0].initialization["overrides"]) == {}
    explicit_true = resolve("linear(4, trainable=True)", **kwargs)
    assert explicit_true.nodes[0].trainability["default"] is True
    assert explicit_true.semantic_digest != default.semantic_digest


def test_constant_values_are_normalized_to_float32_in_both_frontends():
    kwargs = {"input_shape": ("B", 8), "output_shape": ("B", 4)}
    first = resolve("linear(4, init={'weight': 0.1})", **kwargs)
    second = resolve_callable(lambda x: ops.linear(4, init={"weight": 0.1}), **kwargs)
    assert first.semantic_digest == second.semantic_digest
    expected = struct.unpack("f", struct.pack("f", 0.1))[0]
    assert first.nodes[0].initialization["overrides"]["weight"] == expected
    integer = resolve("linear(4, init={'weight': 0})", **kwargs)
    floating = resolve("linear(4, init={'weight': 0.0})", **kwargs)
    assert integer.semantic_digest == floating.semantic_digest


def test_signed_zero_survives_frontend_normalization_and_plan_roundtrip():
    config = resolve("linear(init={'weight': -0.0})", **SHAPES)
    native = resolve_callable(lambda x: ops.linear(init={"weight": -0.0}), **SHAPES)
    assert config.semantic_digest == native.semantic_digest
    restored = ResolvedPlan.from_json(config.to_json())
    assert math.copysign(1, restored.nodes[0].initialization["overrides"]["weight"]) == -1


@pytest.mark.parametrize("metadata,code", [
    ({"init": None}, "E_INITIALIZATION"),
    ({"init": 0}, "E_INITIALIZATION"),
    ({"init": [0]}, "E_INITIALIZATION"),
    ({"init": {"weight": True}}, "E_INITIALIZATION"),
    ({"init": {"weight": "zeros"}}, "E_INITIALIZATION"),
    ({"init": {"weight": 1e100}}, "E_INITIALIZATION"),
    ({"init": {"weight.*": 0}}, "E_INITIALIZATION"),
    ({"init": {"": 0}}, "E_INITIALIZATION"),
    ({"init": {"weight..value": 0}}, "E_INITIALIZATION"),
    ({"trainable": None}, "E_TRAINABILITY"),
    ({"trainable": 0}, "E_TRAINABILITY"),
    ({"trainable": "false"}, "E_TRAINABILITY"),
    ({"trainable": {"weight": 0}}, "E_TRAINABILITY"),
    ({"trainable": {"weight": None}}, "E_TRAINABILITY"),
    ({"trainable": {"*": False}}, "E_TRAINABILITY"),
])
def test_invalid_settings_have_matching_diagnostics_and_do_not_mutate_capture(metadata, code):
    arguments = ", ".join(f"{name}={value!r}" for name, value in metadata.items())
    with pytest.raises(HNDLError, match=code) as error:
        capture_config(f"\n    linear(4, {arguments})\n", **SHAPES)
    assert (error.value.line, error.value.column) == (2, 5)

    def author(x):
        ops.linear(4)
        with pytest.raises(HNDLError, match=code):
            ops.linear(3, name="failed", **metadata)
        ops.relu(name="failed")  # Failed calls must not reserve their names.

    graph = capture_callable(author, **SHAPES)
    assert [node.id for node in graph.nodes] == ["n0", "failed"]
    assert graph.nodes[1].inputs["x"] == "node:n0/out"


def test_invalid_metadata_is_rejected_before_current_binding():
    def author(x):
        a, b = ops.split(4)
        with pytest.raises(HNDLError, match="E_INITIALIZATION"):
            ops.linear(4, init=None)
        with pytest.raises(HNDLError, match="E_CURRENT"):
            ops.linear(4)
        return a

    graph = capture_callable(author, **SHAPES)
    assert len(graph.nodes) == 1
    assert graph.output_ref == "node:n0/first"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_native_nonfinite_initializers_fail_before_creating_nodes(value):
    def author(x):
        with pytest.raises(HNDLError, match="E_INITIALIZATION"):
            ops.linear(4, init={"weight": value})
        return ops.linear(2)

    graph = capture_callable(author, **SHAPES)
    assert [node.id for node in graph.nodes] == ["n0"]


def test_custom_arguments_stay_separate_from_exact_nested_parameter_settings():
    registry = Registry.builtins()

    @registry.operator("block", identity="tests.block", summary="Gain block.", shape="x[B, ...] -> out[B, ...]",
                       args={"gain": Arg(float, 1.0, help="Multiplier.")})
    class Block(nn.Module):
        def __init__(self, gain):
            super().__init__()
            self.gain = gain
            self.projection = nn.Sequential(nn.Linear(8, 8))
            self._scale = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return self.projection(x) * self.gain * self._scale

    source = '''
    block(0.5, init={"projection.0.weight": 0, "_scale": 1},
          trainable={"projection.0.weight": False}, name="configured")
    '''

    def author(x):
        return registry.ops.block(0.5, init={"projection.0.weight": 0, "_scale": 1},
                                  trainable={"projection.0.weight": False}, name="configured")

    kwargs = dict(registry=registry, input_shape=("B", 8), output_shape=("B", 8))
    first = resolve(source, **kwargs)
    second = resolve_callable(author, **kwargs)
    assert first.semantic_digest == second.semantic_digest
    node = first.nodes[0]
    assert dict(node.args) == {"gain": 0.5}
    assert dict(node.initialization["overrides"]) == {"projection.0.weight": 0.0, "_scale": 1.0}
    assert dict(node.trainability["overrides"]) == {"projection.0.weight": False}


def test_metadata_maps_are_snapshots_not_shared_mutable_author_objects():
    initialization = {"weight": 0}
    trainability = {"weight": False}

    def author(x):
        return ops.linear(2, init=initialization, trainable=trainability)

    plan = resolve_callable(author, **SHAPES)
    original_digest = plan.semantic_digest
    initialization["weight"] = 1
    trainability["weight"] = True
    assert plan.nodes[0].initialization["overrides"]["weight"] == 0.0
    assert plan.nodes[0].trainability["overrides"]["weight"] is False
    assert plan.semantic_digest == original_digest
    with pytest.raises(TypeError):
        plan.nodes[0].trainability["overrides"]["weight"] = True


@pytest.mark.parametrize("field", ["init", "trainable"])
def test_parameter_map_and_path_limits_preserve_capture_state(field):
    value = 0 if field == "init" else False
    metadata = {field: {f"parameter{i}": value for i in range(257)}}

    def author(x):
        with pytest.raises(HNDLError):
            ops.linear(4, **metadata)
        with pytest.raises(HNDLError):
            ops.linear(4, **{field: {"x" * 257: value}})
        return ops.linear(2)

    assert len(capture_callable(author, **SHAPES).nodes) == 1
    with pytest.raises(HNDLError) as error:
        capture_config(f"linear(4, {field}={metadata[field]!r})", **SHAPES)
    assert (error.value.line, error.value.column) == (1, 1)


@pytest.mark.parametrize("field", ["init", "trainable"])
def test_settings_names_are_reserved_against_custom_scalar_and_input_port_collisions(field):
    registry = Registry.builtins()
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        registry.operator("scalar", identity="tests.scalar", summary="Bad.", shape="x[B, ...] -> out[B, ...]",
                          args={field: Arg(bool, False, help="Reserved name.")})(nn.Identity)
    with pytest.raises(HNDLError, match="E_REGISTRY"):
        registry.operator("port", identity="tests.port", summary="Bad.",
                          shape=f"{field}[B, ...] -> out[B, ...]")(nn.Identity)


@pytest.mark.parametrize("invalid", [
    'linear(init={"weight": relu()})',
    'linear(init={"weight": x.weight})',
    'linear(init=lambda: 0)',
    'linear(init={"weight": 0, "weight": 1})',
    'linear(trainable={"weight": False, "weight": True})',
    # Initializer calls are legal only as a direct value of an init= mapping.
    'linear(init=xavier_uniform())',
    'linear(init=[xavier_uniform()])',
    'linear(init={"weight": [xavier_uniform()]})',
    'linear(init={"weight": {"nested": xavier_uniform()}})',
    'linear(init={"weight": xavier_uniform(gain=relu())})',
    'linear(init={"weight": xavier_uniform(gain=xavier_uniform())})',
    'linear(init={"weight": xavier_uniform(1.0)})',
    'linear(4, trainable={"weight": xavier_uniform()})',
    'linear(4, other={"weight": xavier_uniform()})',
    'linear(4, name=xavier_uniform())',
    'xavier_uniform()',
    'w = xavier_uniform()',
])
def test_metadata_dictionaries_do_not_widen_config_execution_permissions(invalid):
    registry = Registry.builtins()
    calls = []

    def no_lookup(*args):
        calls.append(args)
        raise AssertionError("No operator lookup is allowed before full AST validation")

    registry.get = no_lookup
    with pytest.raises(HNDLError, match="E_SYNTAX"):
        capture_config(f"linear(4)\n{invalid}", registry=registry, **SHAPES)
    assert calls == []


def test_initializer_schemes_agree_across_frontends_and_survive_plan_roundtrip():
    source = '''
    linear(4, init={"weight": xavier_uniform(gain=1.5), "bias": 0.0}, name="body")
    relu()
    linear(init={"weight": truncated_normal(std=0.25), "bias": kaiming_normal(mode="fan_out")},
           trainable={"bias": False}, name="head")
    '''

    def author(x):
        ops.linear(4, init={"weight": xavier_uniform(gain=1.5), "bias": 0.0}, name="body")
        ops.relu()
        ops.linear(init={"weight": truncated_normal(std=0.25), "bias": kaiming_normal(mode="fan_out")},
                   trainable={"bias": False}, name="head")

    config_plan = resolve(source, **SHAPES)
    native_plan = resolve_callable(author, **SHAPES)
    assert config_plan.semantic_digest == native_plan.semantic_digest
    restored = ResolvedPlan.from_json(config_plan.to_json())
    assert restored.semantic_digest == config_plan.semantic_digest
    body = restored.nodes[0]
    assert body.initialization["kind"] == "torch_default@2"
    assert dict(body.initialization["overrides"]) == {
        "weight": {"kind": "xavier_uniform", "gain": 1.5}, "bias": 0.0}
    head = restored.nodes[-1]
    assert dict(head.initialization["overrides"]["bias"]) == {
        "kind": "kaiming_normal", "a": 0.0, "mode": "fan_out", "nonlinearity": "leaky_relu"}
    assert dict(head.initialization["overrides"]["weight"]) == {
        "kind": "truncated_normal", "mean": 0.0, "std": 0.25, "a": -2.0, "b": 2.0}
    # The parameterless relu node keeps the original constant-only version.
    assert restored.nodes[1].initialization == {"kind": "torch_default@1", "overrides": {}}


@pytest.mark.parametrize("call, factory, expected", [
    ("xavier_uniform()", lambda: xavier_uniform(),
     {"kind": "xavier_uniform", "gain": 1.0}),
    ("xavier_normal(gain=0.5)", lambda: xavier_normal(gain=0.5),
     {"kind": "xavier_normal", "gain": 0.5}),
    ("kaiming_uniform()", lambda: kaiming_uniform(),
     {"kind": "kaiming_uniform", "a": 0.0, "mode": "fan_in", "nonlinearity": "leaky_relu"}),
    ('kaiming_normal(a=0.25, mode="fan_out", nonlinearity="relu")',
     lambda: kaiming_normal(a=0.25, mode="fan_out", nonlinearity="relu"),
     {"kind": "kaiming_normal", "a": 0.25, "mode": "fan_out", "nonlinearity": "relu"}),
    ("truncated_normal()", lambda: truncated_normal(),
     {"kind": "truncated_normal", "mean": 0.0, "std": 1.0, "a": -2.0, "b": 2.0}),
    ("normal(mean=-1.5, std=0.125)", lambda: normal(mean=-1.5, std=0.125),
     {"kind": "normal", "mean": -1.5, "std": 0.125}),
    ("uniform(a=-0.5, b=0.5)", lambda: uniform(a=-0.5, b=0.5),
     {"kind": "uniform", "a": -0.5, "b": 0.5}),
    ("orthogonal(gain=2)", lambda: orthogonal(gain=2),
     {"kind": "orthogonal", "gain": 2.0}),
])
def test_every_scheme_matches_its_torch_defaults_in_both_frontends(call, factory, expected):
    kwargs = {"input_shape": ("B", 8), "output_shape": ("B", 4)}
    config_plan = resolve(f'linear(4, init={{"weight": {call}}})', **kwargs)
    scheme = factory()
    native_plan = resolve_callable(lambda x: ops.linear(4, init={"weight": scheme}), **kwargs)
    assert scheme == expected
    assert config_plan.semantic_digest == native_plan.semantic_digest
    assert dict(config_plan.nodes[0].initialization["overrides"]["weight"]) == expected


def test_scheme_arguments_round_to_float32_like_constant_overrides():
    kwargs = {"input_shape": ("B", 8), "output_shape": ("B", 4)}
    plan = resolve('linear(4, init={"weight": normal(std=0.1)})', **kwargs)
    expected = struct.unpack("f", struct.pack("f", 0.1))[0]
    assert plan.nodes[0].initialization["overrides"]["weight"]["std"] == expected
    assert plan.semantic_digest == resolve_callable(
        lambda x: ops.linear(4, init={"weight": normal(std=0.1)}), **kwargs).semantic_digest
    # Integer and float spellings of the same argument are one canonical record.
    assert (resolve('linear(4, init={"weight": uniform(a=-1, b=1)})', **kwargs).semantic_digest
            == resolve('linear(4, init={"weight": uniform(a=-1.0, b=1.0)})', **kwargs).semantic_digest)


def test_constant_only_plans_keep_the_original_initializer_version():
    constants = resolve('linear(4, init={"weight": 0, "bias": 1})', input_shape=("B", 8), output_shape=("B", 4))
    assert constants.nodes[0].initialization["kind"] == "torch_default@1"
    schemes = resolve('linear(4, init={"weight": xavier_uniform(), "bias": 1})',
                      input_shape=("B", 8), output_shape=("B", 4))
    assert schemes.nodes[0].initialization["kind"] == "torch_default@2"
    assert schemes.semantic_digest != constants.semantic_digest


@pytest.mark.parametrize("call", [
    'xavier_uniform(gian=1.0)',
    'xavier_uniform(gain=1.0, mode="fan_in")',
    'orthogonal(a=1.0, b=2.0)',
])
def test_unknown_scheme_keywords_name_the_arguments_the_scheme_accepts(call):
    with pytest.raises(HNDLError, match="E_INITIALIZATION.*it takes") as error:
        capture_config(f'linear(4, init={{"weight": {call}}})', **SHAPES)
    assert (error.value.line, error.value.column) == (1, 27)
    # The trusted frontend keeps torch's exact signatures, so Python rejects it.
    with pytest.raises(TypeError):
        xavier_uniform(gian=1.0)


@pytest.mark.parametrize("call, factory", [
    ('kaiming_normal(mode="fan_sideways")', lambda: kaiming_normal(mode="fan_sideways")),
    ('kaiming_normal(nonlinearity="swish")', lambda: kaiming_normal(nonlinearity="swish")),
    ('kaiming_normal(mode=1.0)', lambda: kaiming_normal(mode=1.0)),
    ('normal(std="wide")', lambda: normal(std="wide")),
    ('normal(mean=True)', lambda: normal(mean=True)),
    ('uniform(a=1e39)', lambda: uniform(a=1e39)),
])
def test_invalid_scheme_arguments_report_initialization_errors_with_a_location(call, factory):
    with pytest.raises(HNDLError, match="E_INITIALIZATION") as error:
        capture_config(f'\n    linear(4, init={{"weight": {call}}})\n', **SHAPES)
    assert error.value.line == 2
    # The trusted frontend rejects the same arguments before any node exists.
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        factory()


def test_initializer_names_cannot_be_registered_or_parsed_as_operator_aliases():
    registry = Registry.builtins()
    with pytest.raises(HNDLError, match="E_REGISTRY.*reserved initializer scheme"):
        registry.operator("normal", identity="tests.normal", summary="Bad.",
                          shape="x[B, ...] -> out[B, ...]")(nn.Identity)
    with pytest.raises(HNDLError, match="E_NAME.*orthogonal, xavier_uniform"):
        _parse("relu()", ("relu", "xavier_uniform", "orthogonal"))


def test_native_frontend_accepts_only_canonical_scheme_mappings():
    for value in ({}, {"kind": "glorot"}, {"kind": "normal", "sigma": 1.0}, {"kind": 4}):
        with pytest.raises(HNDLError, match="E_INITIALIZATION"):
            capture_callable(lambda x: ops.linear(4, init={"weight": value}), **SHAPES)
    graph = capture_callable(lambda x: ops.linear(4, init={"weight": {"kind": "normal", "std": 0.5}}), **SHAPES)
    assert dict(graph.nodes[0].initialization["overrides"]["weight"]) == {
        "kind": "normal", "mean": 0.0, "std": 0.5}
