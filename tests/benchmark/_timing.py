"""Timing helpers shared by the benchmark tests.

The leading underscore keeps pytest from collecting this as a test module.

Measurements come from ``torch.utils.benchmark.Timer`` rather than a
hand-rolled ``time.perf_counter`` loop: it synchronizes CUDA before and after
each replicate and its blocked autorange strategy picks a loop count large
enough that the timer resolution stops mattering.

Verified against torch 2.14: ``blocked_autorange`` returns a
``torch.utils.benchmark.utils.common.Measurement`` whose ``.mean`` and
``.median`` are per-call times in **seconds** (it also carries ``.iqr``,
``.times`` and ``.raw_times``).
"""

import torch.utils.benchmark as torch_benchmark

__all__ = ["measure", "format_measurement"]


def measure(fn, *, min_run_time=0.2):
    """Time ``fn()`` and return a ``torch.utils.benchmark.Measurement``.

    ``min_run_time`` is the wall-clock budget in seconds; the timer keeps
    collecting replicates until it has spent at least that long.
    """
    timer = torch_benchmark.Timer(stmt="fn()", globals={"fn": fn})
    return timer.blocked_autorange(min_run_time=min_run_time)


def format_measurement(name, measurement):
    """One readable line, e.g. ``"resnet cpu: 1.234 ms (median 1.200 ms)"``."""
    mean_ms = measurement.mean * 1e3
    median_ms = measurement.median * 1e3
    return f"{name}: {mean_ms:.3f} ms (median {median_ms:.3f} ms)"
