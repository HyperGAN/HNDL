"""Prepared execution, streaming, and deployment measurements with parity checks.

Run explicitly with ``pytest -m benchmark tests/benchmark/test_prepared_execution.py -s``.
Every measurement prints a machine-readable HNDL_PERF JSON record. Absolute
latency and speedup are observations, never portable pass/fail thresholds.
"""

from contextlib import nullcontext
import gc
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time

import pytest
import torch

from hndl.torch import InferenceCache, network

from ._timing import measure

pytestmark = pytest.mark.benchmark

MLP = "linear(128); relu(); linear(128); relu(); linear(10)"


def _emit(kind, **values):
    print("HNDL_PERF " + json.dumps({"kind": kind, **values}, sort_keys=True), flush=True)


@pytest.fixture(scope="module", autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cpu = next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    _emit("environment", python=platform.python_version(), torch=str(torch.__version__),
          cpu=cpu, intraop_threads=1,
          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    yield
    torch.set_num_threads(threads)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _elapsed(fn, device):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return time.perf_counter() - start, result


def _latencies(cases, min_time):
    rounds = {name: [] for name in cases}
    for iteration in range(3):
        names = list(cases) if iteration % 2 == 0 else list(reversed(cases))
        for name in names:
            result = measure(cases[name], min_run_time=min_time / 3)
            rounds[name].append(result.median * 1e6)
    return {name: round(statistics.median(times), 3) for name, times in rounds.items()}


@pytest.mark.parametrize("batch", [1, 32])
def test_prepared_inference_setup_latency_and_workspace(device, batch, benchmark_min_time):
    model = network(MLP, input_shape=("B", 64), output_shape=("B", 10),
                    device=device, initialization_seed=1).eval()
    x = torch.randn(batch, 64, device=device)
    with torch.inference_mode():
        expected = model(x)
        setup, generated = _elapsed(lambda: model.prepare((x,)), device)
        torch.testing.assert_close(generated(x), expected)
        cases = {"eager": lambda: model(x), "generated": lambda: generated(x)}
        prepared = []
        workspace = {}
        if str(device).startswith("cuda"):
            for clone in (True, False):
                gc.collect()
                torch.cuda.empty_cache()
                before = torch.cuda.memory_reserved(device)
                label = "replay_clone" if clone else "replay_borrowed"
                cold, executor = _elapsed(
                    lambda clone=clone: model.prepare((x,), backend="cuda_graph", clone_outputs=clone,
                                                      max_graphs=1, max_graph_bytes=32 * 1024**2), device)
                torch.testing.assert_close(executor(x), expected)
                _sync(device)
                workspace[label] = {
                    "setup_ms": round(cold * 1e3, 3),
                    "reported_bytes": executor.cache_info["bytes"],
                    "reserved_delta_bytes": torch.cuda.memory_reserved(device) - before,
                    "graphs": executor.cache_info["graphs"],
                }
                assert executor.cache_info["graphs"] == 1
                assert executor.cache_info["bytes"] <= 32 * 1024**2
                cases[label] = lambda executor=executor: executor(x)
                prepared.append(executor)
        timings = _latencies(cases, benchmark_min_time)
        _emit("inference", device=str(device), batch=batch,
              generated_setup_ms=round(setup * 1e3, 3), median_us=timings, workspace=workspace)
        for executor in prepared:
            executor.clear_cache()
            assert executor.cache_info["graphs"] == executor.cache_info["bytes"] == 0


@pytest.mark.parametrize("batch", [1, 32])
def test_prepared_training_forward_backward(device, batch, benchmark_min_time):
    model = network(MLP, input_shape=("B", 64), output_shape=("B", 10),
                    device=device, initialization_seed=1).train()
    x = torch.randn(batch, 64, device=device)
    generated = model.prepare()

    def step(module):
        module.zero_grad(set_to_none=True)
        module(x).square().mean().backward()

    step(model)
    expected = tuple(parameter.grad.clone() for parameter in model.parameters())
    step(generated)
    for actual, reference in zip(model.parameters(), expected):
        torch.testing.assert_close(actual.grad, reference)
    timings = _latencies({"eager": lambda: step(model), "generated": lambda: step(generated)}, benchmark_min_time)
    _emit("training", device=str(device), batch=batch, median_us=timings,
          generated_over_eager=round(timings["generated"] / timings["eager"], 3))


@pytest.mark.cuda
def test_replay_workspace_under_repeated_eviction(device):
    if not str(device).startswith("cuda"):
        pytest.skip("CUDA replay workspace is measured on CUDA")
    model = network(MLP, input_shape=("B", 64), output_shape=("B", 10), device=device).eval()
    budget = 8 * 1024**2
    with torch.inference_mode():
        model(torch.randn(1, 64, device=device))
        gc.collect()
        torch.cuda.empty_cache()
        _sync(device)
        baseline = torch.cuda.memory_reserved(device)
        generated = model.prepare(backend="cuda_graph", clone_outputs=False,
                                  max_graphs=1, max_graph_bytes=budget)
        entries = []
        for batch in range(1, 7):
            generated(torch.randn(batch, 64, device=device))
            gc.collect()
            _sync(device)
            torch.cuda.empty_cache()
            assert generated.cache_info["graphs"] == 1
            assert generated.cache_info["bytes"] <= budget
            entries.append({"batch": batch,
                            "reserved_delta_bytes": torch.cuda.memory_reserved(device) - baseline,
                            "reported_bytes": generated.cache_info["bytes"]})
        # Fixed framework workspace is allowed; another full workspace per
        # evicted entry would violate bounded retention and must be visible.
        assert entries[-1]["reserved_delta_bytes"] <= entries[0]["reserved_delta_bytes"] + budget
        generated.clear_cache()
        gc.collect()
        _sync(device)
        torch.cuda.empty_cache()
        _emit("workspace_eviction", device=str(device), max_graphs=1, max_bytes=budget,
              entries=entries, after_clear_reserved_delta_bytes=torch.cuda.memory_reserved(device) - baseline)


@pytest.mark.parametrize("prefix,width", [(256, 128), (1024, 128), (1024, 512)])
def test_incremental_steps_vs_full_prefix(device, prefix, width):
    steps = 16
    model = network("for _ in range(2):\n    transformer_block(4, causal=True, rope=True)",
                    input_shape=("B", prefix + steps, width), output_shape=("B", prefix + steps, width),
                    device=device, initialization_seed=1).eval()
    tokens = torch.randn(1, prefix + steps, width, device=device)
    # The generated tensor program accepts the growing full prefix. Its
    # original fixed-T HNDL boundary contract is deliberately outside this
    # baseline, which gives recomputation the cheapest ordinary execution.
    full = model.prepare()._export_module
    cache = InferenceCache(max_seq_len=prefix + steps)
    tolerances = {"rtol": 1e-3, "atol": 1e-3} if str(device).startswith("cuda") else {"rtol": 1e-4, "atol": 1e-5}
    with torch.inference_mode():
        expected = full(tokens)[:, prefix:]

        def prefill():
            cache.reset()
            model(tokens[:, :prefix], cache=cache)

        def decode():
            output = []
            for position in range(prefix, prefix + steps):
                output.append(model(tokens[:, position:position + 1], cache=cache))
            return torch.cat(output, dim=1)

        def recompute():
            output = []
            for position in range(prefix, prefix + steps):
                output.append(full(tokens[:, :position + 1])[:, -1:])
            return torch.cat(output, dim=1)

        prefill()
        torch.testing.assert_close(decode(), expected, **tolerances)
        torch.testing.assert_close(recompute(), expected, **tolerances)
        cached, baseline = [], []
        for iteration in range(3):
            # A finite session keeps the public cache capacity and position
            # semantics intact. Prefill is outside the timed token burst.
            prefill()
            if iteration % 2 == 0:
                duration, actual = _elapsed(decode, device)
                cached.append(duration / steps * 1e6)
                duration, reference = _elapsed(recompute, device)
                baseline.append(duration / steps * 1e6)
            else:
                duration, reference = _elapsed(recompute, device)
                baseline.append(duration / steps * 1e6)
                duration, actual = _elapsed(decode, device)
                cached.append(duration / steps * 1e6)
            torch.testing.assert_close(actual, reference, **tolerances)
        _emit("incremental", device=str(device), batch=1, prefix=prefix, width=width, steps=steps,
              cached_step_us=round(statistics.median(cached), 3),
              recompute_step_us=round(statistics.median(baseline), 3),
              cache_bytes=cache.allocated_bytes, max_seq_len=cache.max_seq_len)


def test_native_deployment_setup_and_warm_latency(device, tmp_path, benchmark_min_time):
    from hndl.deployment import export, load
    import torch._inductor as inductor
    from torch._inductor import utils

    if not hasattr(inductor, "aoti_compile_and_package"):
        pytest.skip("AOTInductor package API unavailable")
    if str(device).startswith("cuda") and not os.environ.get("CUDA_HOME"):
        pytest.skip("native CUDA compilation requires configured CUDA_HOME")
    model = network(MLP, input_shape=("B", 64), output_shape=("B", 10),
                    device=device, initialization_seed=1).eval()
    x = torch.randn(1, 64, device=device)
    fresh = getattr(utils, "fresh_inductor_cache", None)
    with fresh() if fresh is not None else nullcontext():
        with torch.inference_mode():
            expected = model(x)
            cold, artifact = _elapsed(lambda: export(model, (x,), tmp_path), device)
            cached, cached_artifact = _elapsed(lambda: export(model, (x,), tmp_path), device)
            loading, deployed = _elapsed(lambda: load(artifact), device)
            assert cached_artifact.cache_hit and cached_artifact.path == artifact.path
            torch.testing.assert_close(deployed(x), expected, rtol=1e-4, atol=1e-5)
            cases = {"eager": lambda: model(x), "native": lambda: deployed(x)}
            if "run_single_threaded" in inspect.signature(inductor.aoti_load_package).parameters:
                serial = load(artifact, run_single_threaded=True)
                torch.testing.assert_close(serial(x), expected, rtol=1e-4, atol=1e-5)
                cases["native_exclusive"] = lambda: serial(x)
            timings = _latencies(cases, benchmark_min_time)
            _emit("deployment", device=str(device), batch=1, fresh_inductor_cache=fresh is not None,
                  export_seconds=round(cold, 3), cache_hit_ms=round(cached * 1e3, 3),
                  load_ms=round(loading * 1e3, 3), median_us=timings,
                  package_bytes=artifact.path.stat().st_size)
