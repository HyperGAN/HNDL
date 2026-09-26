"""Generic checks every built-in operator must pass through its declared examples.

These run the public harness in ``hndl.testing``, the same one a package that
declares its own operators runs, so the built-in suite and the published
harness cannot drift apart. Operator-specific numerics and error codes live
in ``tests/operators/test_<alias>.py``.
"""

import pytest
import torch

from hndl import Registry, testing

REGISTRY = Registry.builtins()


@pytest.mark.parametrize("spec", testing.operator_params(REGISTRY))
def test_declaration_is_complete(spec):
    testing.check_declaration(REGISTRY, spec)


@pytest.mark.parametrize("spec,example", testing.example_params(REGISTRY))
def test_example_resolves_identically_in_both_frontends_and_round_trips(spec, example):
    testing.check_round_trip(REGISTRY, spec, example)


@pytest.mark.parametrize("spec,example", testing.example_params(REGISTRY))
def test_example_builds_runs_and_is_deterministic(spec, example, device):
    testing.check_build_and_run(REGISTRY, spec, example, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", testing.REDUCED_PRECISION)
@pytest.mark.parametrize("spec,example", testing.example_params(REGISTRY))
def test_example_runs_in_reduced_precision_on_cuda(spec, example, dtype):
    testing.check_build_and_run(REGISTRY, spec, example, device="cuda:0", dtype=dtype)


@pytest.mark.parametrize("spec,example", testing.example_params(REGISTRY))
def test_reference_implementation_matches(spec, example, device):
    if not testing.check_reference(REGISTRY, spec, example, device=device):
        pytest.skip(f"{spec.alias} declares no reference implementation")
