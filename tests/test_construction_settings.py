from dataclasses import replace
import inspect
import json
import math
import struct

import pytest

from hndl import Argument, HNDLError, Registry, ShapeRule, preserves_shape
from hndl.resolver import resolve_graph, validate_concrete_plan
from hndl.settings import (normalize_settings, normalize_initialization, normalize_trainability,
                           validate_initialization, validate_trainability)
from hndl.types import Graph, Node, ResolvedPlan, canonical, digest


def plan_with_settings(**metadata):
    initialization, trainability = normalize_settings(**metadata)
    node = Node("layer", "linear@1", {}, {"x": "input:x"},
                initialization=initialization, trainability=trainability)
    return resolve_graph(Graph((node,), ("B", 3), ("B", 2), "node:layer/out"))


def rehash(data):
    semantic = {key: value for key, value in data.items()
                if key not in {"frontend", "artifact_digest", "semantic_digest"}}
    semantic["nodes"] = [{key: value for key, value in node.items()
                          if key not in {"source", "provenance"}} for node in data["nodes"]]
    data["semantic_digest"] = digest(semantic)
    data.pop("artifact_digest", None)
    data["artifact_digest"] = digest(data)
    return canonical(data)


def test_defaults_and_empty_metadata_are_canonical_and_equivalent():
    default = plan_with_settings()
    explicit_empty = plan_with_settings(init={}, trainable={})
    assert default.semantic_digest == explicit_empty.semantic_digest
    assert default.nodes[0].initialization == {"kind": "torch_default@1", "overrides": {}}
    assert default.nodes[0].trainability == {"default": None, "overrides": {}}
    assert default.schema_version == 2 and default.resolution_version == 1
    assert default.to_json() == ResolvedPlan.from_json(default.to_json()).to_json()


def test_metadata_survives_resolution_validation_and_changes_semantic_identity():
    plan = plan_with_settings(init={"weight": 0, "bias": 0.1}, trainable={"weight": False})
    node = plan.nodes[0]
    assert node.initialization["overrides"] == {"weight": 0.0, "bias": struct.unpack("!f", struct.pack("!f", 0.1))[0]}
    assert node.trainability == {"default": None, "overrides": {"weight": False}}
    assert "initialization" not in node.args and "trainable" not in node.args
    validated = validate_concrete_plan(plan)
    assert validated.semantic_digest == plan.semantic_digest
    assert validated.nodes[0].initialization == node.initialization
    assert ResolvedPlan.from_json(plan.to_json()).nodes[0].trainability == node.trainability
    assert plan.semantic_digest != plan_with_settings().semantic_digest
    assert plan_with_settings(trainable=True).semantic_digest != plan_with_settings().semantic_digest
    assert plan_with_settings(trainable=False).nodes[0].trainability == {"default": False, "overrides": {}}
    assert plan_with_settings(init={"weight": 1}).semantic_digest == plan_with_settings(init={"weight": 1.0}).semantic_digest
    assert "trainability" in plan.describe()


def test_settings_are_copied_and_recursively_immutable_and_fields_are_keyword_only():
    raw = {"weight": 2}
    initialization, trainability = normalize_settings(init=raw, trainable={"bias": False})
    node = Node("layer", "linear@1", {}, {"x": "input:x"}, ("out",), None,
                initialization=initialization, trainability=trainability)
    raw["weight"] = 4
    initialization["overrides"]["weight"] = 5
    trainability["overrides"]["bias"] = True
    assert node.initialization["overrides"]["weight"] == 2.0
    assert node.trainability["overrides"]["bias"] is False
    with pytest.raises(TypeError):
        node.initialization["overrides"]["weight"] = 9.0
    with pytest.raises(TypeError):
        node.trainability["overrides"]["bias"] = True
    assert inspect.signature(Node).parameters["initialization"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(Node).parameters["trainability"].kind is inspect.Parameter.KEYWORD_ONLY


def test_float32_rounding_underflow_and_signed_zero_are_explicit():
    overrides = normalize_initialization({"weight": 1e-50, "bias": -0.0})["overrides"]
    assert overrides["weight"] == 0.0
    assert math.copysign(1.0, overrides["bias"]) == -1.0
    positive = plan_with_settings(init={"weight": 0.0})
    negative = plan_with_settings(init={"weight": -0.0})
    assert positive.semantic_digest != negative.semantic_digest
    restored = ResolvedPlan.from_json(negative.to_json())
    assert math.copysign(1.0, restored.nodes[0].initialization["overrides"]["weight"]) == -1.0


@pytest.mark.parametrize("value", [True, None, "zero", float("nan"), float("inf"), -float("inf"), 1e39, 10**400])
def test_invalid_initialization_constants_reject_without_torch(value):
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        normalize_initialization({"weight": value})


@pytest.mark.parametrize("value", [None, 0, 1, "false", {"weight": 0}, {"weight": None}])
def test_trainability_is_strict_boolean_or_boolean_mapping(value):
    with pytest.raises(HNDLError, match="E_TRAINABILITY"):
        normalize_trainability(value)


@pytest.mark.parametrize("path", ["", ".weight", "weight.", "layer..weight", "*.weight", "layer[0].weight", "with space", "a" * 257, 1])
def test_invalid_parameter_paths_reject_for_both_metadata_kinds(path):
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        normalize_initialization({path: 0})
    with pytest.raises(HNDLError, match="E_TRAINABILITY"):
        normalize_trainability({path: False})


def test_dotted_numeric_paths_are_syntax_only_and_parameter_existence_is_backend_owned():
    plan = plan_with_settings(init={"blocks.0.weight": 1}, trainable={"0.bias": False})
    assert plan.nodes[0].initialization["overrides"] == {"blocks.0.weight": 1.0}
    assert ResolvedPlan.from_json(plan.to_json()).semantic_digest == plan.semantic_digest
    for normalizer, value in ((normalize_initialization, 0), (normalize_trainability, False)):
        with pytest.raises(HNDLError, match="E_RESOURCE"):
            normalizer({f"p{i}": value for i in range(257)})


def test_canonical_records_are_validated_not_silently_rewritten():
    for value in (0, 0.1, True, float("inf")):
        with pytest.raises(HNDLError, match="E_INITIALIZATION"):
            validate_initialization({"kind": "torch_default@1", "overrides": {"weight": value}})
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        validate_initialization({"kind": "future@1", "overrides": {}})
    for record in ({}, {"default": None}, {"default": 0, "overrides": {}}, {"default": None, "overrides": {"weight": 0}}):
        with pytest.raises(HNDLError, match="E_TRAINABILITY"):
            validate_trainability(record)
    assert validate_trainability({"default": False, "overrides": {"bias": True}}) == {"default": False, "overrides": {"bias": True}}


def test_saved_settings_missing_or_noncanonical_fail_even_with_recomputed_digests():
    for field in ("initialization", "trainability"):
        data = json.loads(plan_with_settings().to_json())
        del data["nodes"][0][field]
        with pytest.raises(HNDLError, match="E_SCHEMA"):
            ResolvedPlan.from_json(rehash(data))
    data = json.loads(plan_with_settings(init={"weight": 0}).to_json())
    data["nodes"][0]["initialization"]["overrides"]["weight"] = 0
    with pytest.raises(HNDLError, match="E_INITIALIZATION"):
        ResolvedPlan.from_json(rehash(data))
    data = json.loads(plan_with_settings().to_json())
    data["nodes"][0]["trainability"]["default"] = 1
    with pytest.raises(HNDLError, match="E_TRAINABILITY"):
        ResolvedPlan.from_json(rehash(data))


def test_direct_plan_validation_rechecks_settings_after_deliberate_dataclass_bypass():
    plan = plan_with_settings()
    object.__setattr__(plan.nodes[0], "trainability", {"default": 1, "overrides": {}})
    with pytest.raises(HNDLError, match="E_TRAINABILITY"):
        validate_concrete_plan(plan)


def test_schema_one_rejected_with_explicit_reresolve_instruction():
    data = json.loads(plan_with_settings().to_json())
    data["schema_version"] = 1
    with pytest.raises(HNDLError, match="E_STATE_VERSION.*re-resolved"):
        ResolvedPlan.from_json(rehash(data))
    with pytest.raises(HNDLError, match="E_STATE_VERSION.*re-resolve"):
        validate_concrete_plan(replace(plan_with_settings(), schema_version=1))


@pytest.mark.parametrize("metadata", ["init", "trainable"])
def test_construction_metadata_reserved_in_custom_ports_and_arguments(metadata):
    with pytest.raises(HNDLError, match="conflicts"):
        ShapeRule(inputs={metadata: ("B", "F")}, outputs={"out": ("B", "F")})
    with pytest.raises(HNDLError, match="conflicts"):
        ShapeRule(inputs={"x": ("B", "F")}, outputs={metadata: ("B", "F")})
    with pytest.raises(HNDLError, match="conflicts"):
        Registry.builtins().register("custom", identity="example.custom", version=1,
                                     max_state_bytes=0, shape=preserves_shape,
                                     arguments={metadata: Argument(bool, default=False)})

