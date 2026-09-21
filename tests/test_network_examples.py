"""Every authored network in examples/networks resolves, builds, runs, and round-trips."""

import json
from pathlib import Path

import pytest
import torch

from hndl import ResolvedPlan, resolve
from hndl.torch import DTYPES, build, parameter_counts

NETWORKS = Path(__file__).resolve().parents[1] / "examples" / "networks"
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def load(name):
    meta = json.loads((NETWORKS / f"{name}.json").read_text(encoding="utf-8"))
    source = (NETWORKS / f"{name}.hndl").read_text(encoding="utf-8")
    kwargs = {"input_shape": tuple(meta["input_shape"]), "output_shape": tuple(meta["output_shape"])}
    for key in ("input_dtype", "dtype"):
        if key in meta:
            kwargs[key] = meta[key]
    return source, meta, kwargs


def names():
    return sorted(path.stem for path in NETWORKS.glob("*.hndl"))


def test_every_network_has_a_complete_sidecar():
    assert names(), "examples/networks is empty"
    for name in names():
        meta = json.loads((NETWORKS / f"{name}.json").read_text(encoding="utf-8"))
        for key in ("title", "description", "input_shape", "output_shape", "reference"):
            assert meta.get(key), f"{name}.json lacks {key}"


@pytest.mark.parametrize("name", names())
def test_network_resolves_and_round_trips(name):
    source, meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    assert tuple(plan.output_shape) == tuple(meta["output_shape"])
    assert ResolvedPlan.from_json(plan.to_json()).semantic_digest == plan.semantic_digest
    if meta.get("parameters") is not None:
        counts = parameter_counts(plan)
        assert sum(counts.values()) == meta["parameters"], f"{name}: {sum(counts.values()):,} parameters"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", names())
def test_network_builds_and_runs(name, device):
    source, meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    model = build(plan, device=device, initialization_seed=1)
    shape = (2, *kwargs["input_shape"][1:])
    if kwargs.get("input_dtype") in ("int64", "int32"):
        x = torch.randint(0, 8, shape, device=device, dtype=getattr(torch, kwargs["input_dtype"]))
    else:
        x = torch.randn(shape, device=device, dtype=DTYPES[plan.dtype], requires_grad=True)
    output = model(x=x)["output"]
    assert tuple(output.shape) == (2, *kwargs["output_shape"][1:])
    assert output.isfinite().all()
    output.float().square().mean().backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for p in trainable)
    if meta.get("parameters") is not None:
        assert sum(p.numel() for p in model.parameters()) == meta["parameters"]
