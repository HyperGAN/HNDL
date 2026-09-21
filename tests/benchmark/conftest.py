"""Shared fixtures and constants for the benchmark suite.

Every test module in this package is marked with ``pytest.mark.benchmark``
(``pytestmark = pytest.mark.benchmark`` at module level), so the whole suite
can be selected with ``-m benchmark`` or skipped with ``-m "not benchmark"``.
Run with ``-s`` to see the timing lines the tests print.
"""

import os

import pytest
import torch

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

#: Environment variable that overrides the per-measurement time budget.
MIN_TIME_ENV_VAR = "HNDL_BENCHMARK_MIN_TIME"

#: Seconds of wall clock each ``measure()`` call spends when nothing overrides it.
DEFAULT_MIN_TIME = 0.2


@pytest.fixture(scope="session")
def benchmark_min_time():
    """Seconds to pass as ``measure(..., min_run_time=...)``.

    Defaults to :data:`DEFAULT_MIN_TIME`; set ``HNDL_BENCHMARK_MIN_TIME`` to
    shorten a local run or lengthen it for steadier numbers.
    """
    raw = os.environ.get(MIN_TIME_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_MIN_TIME
    try:
        value = float(raw)
    except ValueError:
        pytest.fail(f"{MIN_TIME_ENV_VAR}={raw!r} is not a number")
    if value <= 0:
        pytest.fail(f"{MIN_TIME_ENV_VAR}={raw!r} must be positive")
    return value


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param
