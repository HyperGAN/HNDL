"""Catalog per-inference latency for every network in examples/networks.

This is instrumentation, not a gate: absolute milliseconds depend on the
machine, so the assertions only check that a measurement happened and that the
timed model produced finite outputs. ``tests/test_network_examples.py`` is
where correctness lives.

Run it explicitly (it is excluded from the default selection)::

    pytest -m benchmark tests/benchmark/test_example_networks_latency.py -s

CPU only on purpose: CUDA is absent from most checkouts, and a catalog whose
rows change device from machine to machine is not comparable.
"""

import math

import pytest
import torch

from hndl import resolve
from hndl.torch import DTYPES, build
from tests._network_examples import load, names
from tests.benchmark._timing import format_measurement, measure

pytestmark = pytest.mark.benchmark

DEVICE = "cpu"
BATCH = 2


def _build_eval_model(name):
    """Resolve and build ``name`` on the CPU, in eval mode, with fixed init."""
    source, _meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    model = build(plan, device=DEVICE, initialization_seed=1)
    model.eval()
    return plan, model


def _random_inputs(plan, generator):
    """Random tensors matching ``plan.inputs``, mirroring test_network_examples."""
    inputs = {}
    for port, entry in plan.inputs.items():
        shape = (BATCH, *entry["shape"][1:])
        if entry["dtype"] in ("int64", "int32"):
            inputs[port] = torch.randint(0, 8, shape, device=DEVICE, generator=generator,
                                         dtype=getattr(torch, entry["dtype"]))
        else:
            inputs[port] = torch.randn(shape, device=DEVICE, generator=generator,
                                       dtype=DTYPES[plan.dtype])
    return inputs


@pytest.mark.parametrize("name", names())
def test_example_network_inference_latency(name, benchmark_min_time):
    plan, model = _build_eval_model(name)
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    inputs = _random_inputs(plan, generator)
    outputs = {}

    def forward():
        with torch.no_grad():
            outputs.update(model(**inputs))

    forward()  # warm up, and give the sanity check something to look at
    assert set(outputs) == set(plan.outputs)

    measurement = measure(forward, min_run_time=benchmark_min_time)

    assert math.isfinite(measurement.mean) and measurement.mean > 0, f"{name}: no usable timing"
    assert math.isfinite(measurement.median) and measurement.median > 0, f"{name}: no usable timing"
    for port, tensor in outputs.items():
        assert tensor.isfinite().all(), f"{name}: output {port!r} went non-finite while timing"

    print(format_measurement(f"{name} [{DEVICE}]", measurement))
