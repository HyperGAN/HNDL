"""Plans saved by released HNDL versions keep loading, building and digesting.

The fixtures under ``tests/fixtures/plans_0_6_0`` were written by the 0.6.0
release; ``generate.py`` there records how. Adding an optional argument to an
existing operator must not break them (see ``Arg(since=...)``).
"""

import json
from pathlib import Path

import pytest
import torch

from hndl import Arg, HNDLError, ResolvedPlan, resolve
from hndl.torch import build
from hndl.types import digest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plans_0_6_0"
CASES = sorted(path.stem for path in FIXTURES.glob("*.json"))


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_fixtures_cover_the_operators_that_gained_arguments():
    ops = {node["op"] for name in CASES for node in json.loads(_fixture(name)["plan"])["nodes"]}
    assert {"linear@1", "attention@1", "feed_forward@1"} <= ops


@pytest.mark.parametrize("name", CASES)
def test_saved_0_6_0_plan_loads_and_builds(name):
    record = _fixture(name)
    plan = ResolvedPlan.from_json(record["plan"])
    # Loading and saving again reproduces the 0.6.0 bytes exactly.
    assert plan.to_json() == record["plan"]
    model = build(plan, device="cpu", initialization_seed=0)
    batch = 2
    shape = (batch, *record["input_shape"][1:])
    if record.get("input_dtype") == "int64":
        x = torch.randint(0, 256, shape)
    else:
        x = torch.randn(shape)
    assert tuple(model(x=x)["output"].shape) == (batch, *record["output_shape"][1:])


@pytest.mark.parametrize("name", CASES)
def test_resolving_the_same_source_matches_the_0_6_0_digest(name):
    record = _fixture(name)
    extra = {"input_dtype": record["input_dtype"]} if "input_dtype" in record else {}
    plan = resolve(record["source"], input_shape=tuple(record["input_shape"]),
                   output_shape=tuple(record["output_shape"]), **extra)
    saved = json.loads(record["plan"])
    assert plan.semantic_digest == saved["semantic_digest"]
    assert plan.to_json() == record["plan"]


def test_added_argument_is_kept_only_when_it_differs_from_its_default():
    shapes = {"input_shape": ("B", 4), "output_shape": ("B", 8)}
    omitted = resolve("linear(8)", **shapes)
    explicit_default = resolve("linear(8, equalized=False)", **shapes)
    used = resolve("linear(8, equalized=True)", **shapes)
    assert "equalized" not in omitted.nodes[0].args
    assert "equalized" not in omitted.nodes[0].provenance
    assert "equalized" not in omitted.nodes[0].source["argument_origins"]
    assert explicit_default.semantic_digest == omitted.semantic_digest
    assert used.nodes[0].args["equalized"] is True
    assert used.nodes[0].provenance["equalized"] == "explicit"
    assert used.semantic_digest != omitted.semantic_digest
    restored = ResolvedPlan.from_json(used.to_json())
    assert restored.semantic_digest == used.semantic_digest
    assert build(restored, device="cpu")["n0"].equalized is True
    assert build(ResolvedPlan.from_json(omitted.to_json()), device="cpu")["n0"].equalized is False


def test_ties_participate_in_identity_and_migrate_from_legacy_digest():
    shape = {"input_shape": ("B", 8), "output_shape": ("B", 8)}
    source = 'first = linear(8, name="a")\nlinear(8, name="b"{tie})'
    untied = resolve(source.format(tie=""), **shape)
    tied = resolve(source.format(tie=', tie={"weight": "first.weight"}'), **shape)
    explicit = resolve(source.format(tie=', tie={"weight": "a.weight"}'), **shape)
    assert explicit.semantic_digest == tied.semantic_digest != untied.semantic_digest
    assert untied.diff(tied).modified_nodes["b"]["tie"] == ({}, {"weight": "a.weight"})
    assert ResolvedPlan.from_json(tied.to_json()).to_json() == tied.to_json()

    legacy = json.loads(tied.to_json())
    legacy.pop("artifact_digest")
    legacy["semantic_digest"] = untied.semantic_digest
    legacy["artifact_digest"] = digest(legacy)
    with pytest.warns(UserWarning, match="digest to include parameter ties"):
        restored = ResolvedPlan.from_json(json.dumps(legacy))
    assert restored.semantic_digest == tied.semantic_digest
    model = build(restored, device="cpu")
    assert model["a"].weight is model["b"].weight

    legacy.pop("artifact_digest")
    legacy["semantic_digest"] = "0" * 64
    legacy["artifact_digest"] = digest(legacy)
    with pytest.raises(HNDLError, match="semantic_digest"):
        ResolvedPlan.from_json(json.dumps(legacy))


def test_since_requires_a_release_and_a_default():
    assert Arg(bool, False, since="0.7.0", help="Flag.").since == "0.7.0"
    with pytest.raises(HNDLError, match="needs a default"):
        Arg(int, since="0.7.0", help="Width.")
    with pytest.raises(HNDLError, match="release"):
        Arg(bool, False, since="next", help="Flag.")
