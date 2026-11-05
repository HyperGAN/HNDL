"""Native deployment round trips and cache invalidation behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import torch
import torch._inductor as inductor

from hndl import HNDLError, Registry
from hndl.deployment import export, inspect_artifact, load
from hndl.torch import network


def mlp():
    return network("linear(8); relu(); linear(2)", input_shape=("B", 4),
                   output_shape=("B", 2), initialization_seed=7, device="cpu").eval()


@pytest.fixture
def fake_compiler(monkeypatch):
    """Keep storage/invalidation tests independent of native compiler latency."""
    calls = []

    def compile_package(program, *, package_path, inductor_configs):
        calls.append(program)
        # PyTorch 2.6 mutates the config it receives; the manifest must not.
        inductor_configs["aot_inductor.package"] = True
        Path(package_path).write_bytes(b"native-package-for-cache-tests")

    monkeypatch.setattr(inductor, "aoti_compile_and_package", compile_package)
    return calls


def test_import_and_manifest_inspection_do_not_import_torch(tmp_path):
    script = (
        "import sys\n"
        "import hndl.deployment\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_repeated_export_hits_cache_but_weight_updates_recompile(tmp_path, fake_compiler):
    model, x = mlp(), torch.randn(2, 4)
    first = export(model, (x,), tmp_path)
    second = export(model, {"x": x.clone()}, tmp_path)
    assert second.cache_hit and second.path == first.path and len(fake_compiler) == 1
    with torch.no_grad():
        model[0].weight.add_(0.1)
    third = export(model, (x,), tmp_path)
    assert not third.cache_hit and third.cache_key != first.cache_key
    assert len(fake_compiler) == 2


def test_non_state_python_settings_and_forward_code_invalidate(tmp_path, fake_compiler):
    registry = Registry.builtins()

    @registry.operator("gain", identity="test.deployment.gain", summary="Gain setting.",
                       shape="x[B, ...] -> out[B, ...]")
    class Gain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = 2.0
            self.register_buffer("offset", torch.tensor(1.0), persistent=False)

        def forward(self, x):
            return x * self.gain + self.offset

    model = network("gain()", input_shape=("B", 4), output_shape=("B", 4),
                    registry=registry, device="cpu").eval()
    x = torch.randn(2, 4)
    first = export(model, (x,), tmp_path)
    model[0].gain = 3.0
    second = export(model, (x,), tmp_path)
    model[0].offset.add_(2.0)
    third = export(model, (x,), tmp_path)
    Gain.forward = lambda self, x: x * self.gain + self.offset + 4
    fourth = export(model, (x,), tmp_path)
    assert len({a.cache_key for a in (first, second, third, fourth)}) == 4
    assert len(fake_compiler) == 4


def test_input_stride_and_compile_options_select_distinct_packages(tmp_path, fake_compiler):
    model, x = mlp(), torch.randn(2, 4)
    strided = torch.empty(2, 8)[:, ::2].copy_(x)
    first = export(model, (x,), tmp_path)
    second = export(model, (strided,), tmp_path)
    third = export(model, (x,), tmp_path, inductor_configs={"max_autotune": True})
    assert len({a.cache_key for a in (first, second, third)}) == 3
    assert inspect_artifact(third)["specification"]["inductor_configs"] == {"max_autotune": True}


def test_mutable_imported_setting_changes_captured_graph(tmp_path, fake_compiler):
    registry = Registry.builtins()
    settings = types.ModuleType("deployment_example_settings")
    settings.gain = 2.0

    @registry.operator("configured", identity="test.deployment.configured", summary="External setting.",
                       shape="x[B, ...] -> out[B, ...]")
    class Configured(torch.nn.Module):
        def forward(self, x):
            return x * settings.gain

    model = network("configured()", input_shape=("B", 4), output_shape=("B", 4),
                    registry=registry, device="cpu").eval()
    x = torch.randn(2, 4)
    first = export(model, (x,), tmp_path)
    settings.gain = 3.0
    second = export(model, (x,), tmp_path)
    assert first.cache_key != second.cache_key and len(fake_compiler) == 2


def test_damaged_cache_entry_is_rebuilt(tmp_path, fake_compiler):
    model, x = mlp(), torch.randn(2, 4)
    first = export(model, (x,), tmp_path)
    first.path.write_bytes(b"truncated")
    with pytest.raises(HNDLError, match="checksum"):
        load(first)
    second = export(model, (x,), tmp_path)
    assert not second.cache_hit and len(fake_compiler) == 2


def test_cache_limits_evict_old_entries_and_reject_oversized_packages(tmp_path, fake_compiler):
    model = mlp()
    first = export(model, (torch.randn(1, 4),), tmp_path, max_entries=1)
    second = export(model, (torch.randn(2, 4),), tmp_path, max_entries=1)
    assert not first.path.exists() and second.path.exists()
    assert len(list(tmp_path.glob("*/model.pt2"))) == 1
    with pytest.raises(HNDLError, match="max_bytes"):
        export(model, (torch.randn(2, 4),), tmp_path, max_bytes=1)
    with pytest.raises(HNDLError, match="max_bytes"):
        export(model, (torch.randn(3, 4),), tmp_path, max_bytes=1)
    assert len(list(tmp_path.glob("*/model.pt2"))) == 1
    assert not list(tmp_path.glob(".hndl-build-*"))


def test_compilation_failure_leaves_no_visible_package(tmp_path, fake_compiler, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("compiler failure")

    monkeypatch.setattr(inductor, "aoti_compile_and_package", fail)
    with pytest.raises(RuntimeError, match="compiler failure"):
        export(mlp(), (torch.randn(2, 4),), tmp_path)
    assert not list(tmp_path.glob("*/model.pt2"))
    assert not list(tmp_path.glob(".hndl-build-*"))


def test_eval_inputs_and_cache_limits_are_checked(tmp_path, fake_compiler):
    model, x = mlp(), torch.randn(2, 4)
    model.train()
    with pytest.raises(HNDLError, match="eval"):
        export(model, (x,), tmp_path)
    model.eval()
    with pytest.raises(HNDLError, match="detach"):
        export(model, (x.requires_grad_(),), tmp_path)
    with pytest.raises(HNDLError, match="positive"):
        export(model, (x.detach(),), tmp_path, max_entries=0)
    with pytest.raises(HNDLError, match="exactly"):
        export(model, {"wrong": x.detach()}, tmp_path)
    assert not fake_compiler


def test_manifest_tampering_is_rejected(tmp_path, fake_compiler):
    artifact = export(mlp(), (torch.randn(2, 4),), tmp_path)
    manifest = json.loads(artifact.manifest_path.read_text())
    manifest["specification"]["inputs"][0]["shape"][0] = 99
    artifact.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(HNDLError, match="digest"):
        inspect_artifact(artifact)


@pytest.mark.parametrize("named", [False, True])
def test_native_cpu_compile_load_and_input_guards(tmp_path, monkeypatch, named):
    if not hasattr(inductor, "aoti_compile_and_package"):
        pytest.skip("this PyTorch build has no AOTInductor packaging")
    if shutil.which("g++") is None:
        pytest.skip("AOTInductor CPU compilation requires a C++ compiler")
    if named:
        model = network("h = add(x, y); out = linear(h, 2)",
                        input_shape={"x": ("B", 4), "y": ("B", 4)},
                        output_shape={"out": ("B", 2), "h": ("B", 4)}, device="cpu").eval()
        inputs = (torch.randn(2, 4), torch.randn(2, 4))
    else:
        model, inputs = mlp(), (torch.randn(2, 4),)
    with torch.no_grad():
        expected = model(*inputs)
    artifact = export(model, inputs, tmp_path)
    deployed = load(artifact)
    with monkeypatch.context() as patch:
        # Loaded packages must be independent of the Python graph interpreter.
        patch.setattr(type(model), "_execute", lambda *args, **kwargs: pytest.fail("Python runtime was used"))
        actual = deployed(**dict(zip(model.plan.inputs, inputs)))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(HNDLError, match="must match"):
        deployed(*(value[:1] for value in inputs))
    with pytest.raises(HNDLError, match="must match"):
        deployed(*(value.double() for value in inputs))
    strided = tuple(torch.empty(2, 8)[:, ::2].copy_(value) for value in inputs)
    with pytest.raises(HNDLError, match="must match"):
        deployed(*strided)
    with pytest.raises(HNDLError, match="detach"):
        deployed(*(value.clone().requires_grad_() for value in inputs))
    with pytest.raises(HNDLError, match="supplied twice"):
        deployed(*inputs, **{next(iter(model.plan.inputs)): inputs[0]})
    with pytest.raises(HNDLError, match="expects inputs"):
        deployed()

    # Re-check compatibility before native loading, rather than invoking an
    # incompatible binary and depending on a compiler/runtime failure.
    import hndl.deployment as deployment
    environment = deployment._environment
    monkeypatch.setattr(deployment, "_environment", lambda device: {**environment(device), "torch": "different"})
    with pytest.raises(HNDLError, match="runtime or hardware"):
        load(artifact)


@pytest.mark.cuda
def test_native_cuda_compile_load_and_cache_reuse(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA deployment requires a CUDA device")
    if not os.environ.get("CUDA_HOME"):
        pytest.skip("native CUDA compilation requires a configured CUDA toolkit (CUDA_HOME)")
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    model = network("linear(128); relu(); linear(10)", input_shape=("B", 64),
                    output_shape=("B", 10), device=device).eval()
    x = torch.randn(2, 64, device=device)
    with torch.no_grad():
        expected = model(x)
    artifact = export(model, (x,), tmp_path)
    actual = load(artifact)(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert actual.device == x.device
    assert export(model, (x,), tmp_path).cache_hit
