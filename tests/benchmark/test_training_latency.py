"""Benchmark forward + backward training step latency across example networks.

Tests that training graphs execute efficiently and that backpropagation
computes finite gradients across parameters.
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


def _build_train_model(name, device="cpu"):
    """Resolve and build ``name`` on the device, in training mode, with fixed init."""
    source, _meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    model = build(plan, device=device, initialization_seed=1)
    model.train()
    return plan, model


@pytest.mark.parametrize("name", names())
def test_example_network_training_latency(name, benchmark_min_time, device):
    plan, model = _build_train_model(name, device=device)
    generator = torch.Generator(device=device).manual_seed(1)
    inputs = random_inputs(plan, device=device, batch_size=BATCH, generator=generator)

    # Convert non-floating / integer inputs if needed; require grad on floating inputs
    floating_inputs = {}
    for k, v in inputs.items():
        if v.is_floating_point():
            floating_inputs[k] = v.clone().detach().requires_grad_(True)
        else:
            floating_inputs[k] = v

    def train_step():
        model.zero_grad(set_to_none=True)
        outputs = model(**floating_inputs)
        loss = sum(out.sum() for out in outputs.values() if out.is_floating_point())
        loss.backward()

    # Warm-up pass
    train_step()

    # Verify gradients reached at least one parameter
    has_grad = any(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert has_grad or not list(model.parameters()), f"{name}: no finite gradients produced"

    measurement = measure(train_step, min_run_time=benchmark_min_time)
    print(format_measurement(f"training {name} [{device}]", measurement))

    assert math.isfinite(measurement.mean) and measurement.mean > 0, f"{name}: no usable timing"
    assert math.isfinite(measurement.median) and measurement.median > 0, f"{name}: no usable timing"
