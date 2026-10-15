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
from hndl.torch import build
from tests._network_examples import load, names, random_inputs
from tests.benchmark._timing import format_measurement, measure

pytestmark = pytest.mark.benchmark

BATCH = 2


def _build_eval_model(name, device="cpu"):
    """Resolve and build ``name`` on the device, in eval mode, with fixed init."""
    source, _meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    model = build(plan, device=device, initialization_seed=1)
    model.eval()
    return plan, model


@pytest.mark.parametrize("name", names())
def test_example_network_inference_latency(name, benchmark_min_time, device):
    plan, model = _build_eval_model(name, device=device)
    generator = torch.Generator(device=device).manual_seed(1)
    inputs = random_inputs(plan, device=device, batch_size=BATCH, generator=generator)
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

    print(format_measurement(f"{name} [{device}]", measurement))

