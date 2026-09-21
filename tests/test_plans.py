"""Persistence validates concrete graph data, not just its checksum."""

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hndl.errors import HNDLError
from hndl.registry import Registry, preserves_shape
from hndl.resolver import resolve_graph, validate_concrete_plan
from hndl.types import Graph, Node, ResolvedPlan, canonical, digest


def mlp_plan():
    return resolve_graph(Graph(
        nodes=(Node("hidden", "linear@1", {"out_features": 4}, {"x": "input:x"}),
               Node("head", "linear@1", {}, {"x": "node:hidden/out"})),
        input_shape=("B", 3), output_shape=("B", 2), output_ref="node:head/out"))


def test_canonical_encoding_vector():
    value = {"b": (2, 3), "a": "✓"}
    assert canonical(value) == '{"a":"✓","b":[2,3]}'
    assert digest(value) == "afc375fca79ef9ed177bab6269d54c2e2b50bcc6d7a0a670a4cd64489e85ebfa"


def rehash(data):
    # An attacker can recompute checksums; shape/state consistency must still
    # be checked without trusting stored declarations.
    semantic = {key: value for key, value in data.items()
                if key not in {"frontend", "artifact_digest", "semantic_digest"}}
    semantic["nodes"] = [{key: value for key, value in node.items()
                          if key not in {"source", "provenance"}}
                         for node in data["nodes"]]
    data["semantic_digest"] = digest(semantic)
    data.pop("artifact_digest", None)
    data["artifact_digest"] = digest(data)
    return canonical(data)


def test_resolved_records_are_recursively_immutable():
    plan = mlp_plan()
    with pytest.raises(FrozenInstanceError):
        plan.dtype = "float64"
    with pytest.raises(TypeError):
        plan.nodes[0].args["out_features"] = 8
    with pytest.raises(TypeError):
        plan.nodes[0].input_shapes["x"] = ("B", 7)
    with pytest.raises(TypeError):
        plan.nodes[0].output_shapes["out"][1] = 8


def test_plan_round_trip_is_canonical_and_retains_identity():
    plan = mlp_plan()
    encoded = plan.to_json()
    restored = ResolvedPlan.from_json(encoded)
    assert restored.to_json() == encoded
    assert restored.semantic_digest == plan.semantic_digest
    assert len(plan.semantic_digest) == 64
    assert restored.state_bytes == (3 * 4 + 4 + 4 * 2 + 2) * 4
    assert "Semantic digest" in restored.describe()


def test_frontend_and_source_metadata_affect_artifact_not_semantics():
    plan = mlp_plan()
    annotated = replace(plan, frontend="python_callable@1", nodes=(
        replace(plan.nodes[0], source={"line": 10, "column": 3}), plan.nodes[1]))
    assert annotated.semantic_digest == plan.semantic_digest
    assert json.loads(annotated.to_json())["artifact_digest"] != json.loads(plan.to_json())["artifact_digest"]
    changed = resolve_graph(Graph(
        (Node("renamed", "linear@1", {"out_features": 4}, {"x": "input:x"}),
         Node("head", "linear@1", {}, {"x": "node:renamed/out"})),
        ("B", 3), ("B", 2), "node:head/out"))
    assert changed.semantic_digest != plan.semantic_digest


def test_corruption_and_duplicate_json_fields_rejected():
    data = json.loads(mlp_plan().to_json())
    data["nodes"][0]["args"]["out_features"] = 5
    with pytest.raises(HNDLError, match="E_INTEGRITY"):
        ResolvedPlan.from_json(json.dumps(data))
    with pytest.raises(HNDLError, match="E_SCHEMA"):
        ResolvedPlan.from_json('{"schema_version":1,"schema_version":1}')
    with pytest.raises(HNDLError, match="E_SCHEMA"):
        ResolvedPlan.from_json('{"bad":NaN}')


@pytest.mark.parametrize("field,value", [
    ("state_bytes", 0),
    ("output_shapes", {"out": ["B", 999]}),
    ("input_shapes", {"x": ["B", 999]}),
])
def test_rehashed_forged_plan_rejected(field, value):
    data = json.loads(mlp_plan().to_json())
    data["nodes"][0][field] = value
    with pytest.raises(HNDLError, match="E_INTEGRITY"):
        ResolvedPlan.from_json(rehash(data))


def test_persisted_omission_cannot_be_silently_reinferred():
    data = json.loads(mlp_plan().to_json())
    del data["nodes"][1]["args"]["out_features"]
    with pytest.raises(HNDLError, match="E_INTEGRITY"):
        ResolvedPlan.from_json(rehash(data))


@pytest.mark.parametrize("field,value", [
    ("schema_version", 3), ("resolution_version", 2),
    ("schema_version", True), ("resolution_version", True),
])
def test_unknown_and_boolean_versions_rejected(field, value):
    data = json.loads(mlp_plan().to_json())
    data[field] = value
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        ResolvedPlan.from_json(rehash(data))


def test_state_versions_and_allocation_bounds_checked_on_restore():
    data = json.loads(mlp_plan().to_json())
    data["nodes"][0]["state_version"] = 2
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        ResolvedPlan.from_json(rehash(data))
    with pytest.raises(HNDLError, match="E_RESOURCE"):
        ResolvedPlan.from_json(mlp_plan().to_json(), limits={"max_state_bytes": 1})


def test_custom_plan_requires_explicit_exact_registration():
    registry = Registry.builtins()
    registry.register("smooth", identity="example.smooth", version=1,
                      shape=preserves_shape, max_state_bytes=0)
    plan = resolve_graph(Graph((Node("n0", "example.smooth@1", {}, {"x": "input:x"}),),
                               ("B", 3), ("B", 3), "node:n0/out"), registry)
    with pytest.raises(HNDLError, match="E_STATE_VERSION"):
        ResolvedPlan.from_json(plan.to_json())
    assert ResolvedPlan.from_json(plan.to_json(), registry=registry).semantic_digest == plan.semantic_digest


def test_validation_rejects_changed_shape_before_backend_build():
    plan = mlp_plan()
    forged = replace(plan, nodes=(replace(plan.nodes[0], state_bytes=0), plan.nodes[1]))
    with pytest.raises(HNDLError, match="E_INTEGRITY"):
        validate_concrete_plan(forged)


def test_fresh_process_restore_without_torch(tmp_path):
    path = tmp_path / "plan.json"
    plan = mlp_plan()
    path.write_text(plan.to_json())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    script = '''
import sys
from pathlib import Path
from hndl.types import ResolvedPlan
plan = ResolvedPlan.from_json(Path(sys.argv[1]).read_text())
assert "torch" not in sys.modules
print(plan.semantic_digest)
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == plan.semantic_digest
