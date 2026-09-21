"""Does ``torch.compile`` work on a built hndl graph, and does it pay off?

Nothing in hndl itself calls ``torch.compile``: :func:`hndl.torch.build` hands
back an ordinary ``nn.Module``, so compiling it is something a *user* does. This
module is the experiment that checks whether that works transparently for each
authored network in ``examples/networks`` and, when it does, what the inference
speedup actually is.

A network that dynamo cannot trace is **skipped, not failed** -- the point is to
record today's compatibility, not to demand it. Inductor also needs a working
C/C++ toolchain *and* the CPython development headers; a machine without them
(no ``Python.h``) skips every case with the compiler's own error as the reason.
Run with ``-s`` to read the per-network report:

    pytest -m benchmark tests/benchmark/test_compile_compat.py -q -s
"""

import pytest
import torch
import torch._dynamo

from hndl import resolve
from hndl.torch import DTYPES, build
from tests._network_examples import load, names

from ._timing import format_measurement, measure

pytestmark = pytest.mark.benchmark

#: Eager and compiled kernels may pick different reduction orders.
TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}

#: Dynamo errors are famously long; keep the skip reason readable.
MAX_REASON = 400

DEVICE = "cpu"


def _random_inputs(plan):
    """One batch of random inputs matching ``plan.inputs``, batch size 2."""
    inputs = {}
    for name, entry in plan.inputs.items():
        shape = (2, *entry["shape"][1:])
        if entry["dtype"] in ("int64", "int32"):
            inputs[name] = torch.randint(0, 8, shape, device=DEVICE, dtype=getattr(torch, entry["dtype"]))
        else:
            inputs[name] = torch.randn(shape, device=DEVICE, dtype=DTYPES[plan.dtype])
    return inputs


def _truncate(text):
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_REASON else text[:MAX_REASON] + "..."


@pytest.mark.parametrize("name", names())
def test_torch_compile_matches_eager(name, benchmark_min_time):
    # Guards and compiled artifacts are cached per process; clear them so one
    # network's compilation cannot influence the next parametrized case.
    torch._dynamo.reset()

    source, _meta, kwargs = load(name)
    plan = resolve(source, **kwargs)
    model = build(plan, device=DEVICE, initialization_seed=1).eval()
    inputs = _random_inputs(plan)

    with torch.no_grad():
        eager_output = model(**inputs)

    compiled = torch.compile(model)
    try:
        with torch.no_grad():
            compiled_output = compiled(**inputs)
    except Exception as exc:  # noqa: BLE001 - any dynamo/inductor failure is a "no"
        print(f"\n{name}: compiles: no ({type(exc).__name__})")
        pytest.skip(f"torch.compile unsupported for {name}: {_truncate(exc)}")

    if isinstance(eager_output, dict):
        assert set(compiled_output) == set(eager_output)
        for key in eager_output:
            torch.testing.assert_close(compiled_output[key], eager_output[key], **TOLERANCE)
    else:
        torch.testing.assert_close(compiled_output, eager_output, **TOLERANCE)

    with torch.no_grad():
        eager = measure(lambda: model(**inputs), min_run_time=benchmark_min_time)
        compiled_timing = measure(lambda: compiled(**inputs), min_run_time=benchmark_min_time)

    speedup = eager.mean / compiled_timing.mean
    print()
    print(format_measurement(f"{name} eager   ", eager))
    print(format_measurement(f"{name} compiled", compiled_timing))
    print(f"{name}: compiles: yes, speedup: {speedup:.2f}x")
