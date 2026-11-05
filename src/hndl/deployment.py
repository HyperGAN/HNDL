"""Content-addressed, ahead-of-time inference packages.

Importing this module does not import PyTorch. Compilation and loading are
explicit operations; the resulting packages contain native executable code.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import types

from .errors import HNDLError

__all__ = ["DeploymentArtifact", "DeploymentModel", "export", "inspect_artifact", "load"]

_SCHEMA = 1
_DEFAULT_MAX_BYTES = 2 * 1024**3
_MODULE_INTERNALS = frozenset({
    "_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
    "_backward_pre_hooks", "_backward_hooks", "_is_full_backward_hook",
    "_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
    "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_state_dict_hooks",
    "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
    "_compiled_call_impl",
})
# These are derived tables, not operator settings. Unknown custom caches are
# deliberately included: a custom operator may use them as semantic state.
_DERIVED_TABLES = {
    "hndl.operators.attention": {"_rope_cache", "_mask_cache", "_causal_cache"},
    "hndl.operators.cross_attention": {"_mask_cache"},
    "hndl.operators.rope": {"_cache"},
    "hndl.operators.latent_attention": {"_cache"},
}
_PREPARED_METADATA = frozenset({
    "_hndl_packed_layouts", "_hndl_packing_blocked", "_hndl_attention_masks",
    "_hndl_attention_backend", "_hndl_attention_fallback_reason",
    "_hndl_packing_owner", "_hndl_packing_module_id",
})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


class _HashWriter:
    def __init__(self):
        self.hash = hashlib.sha256()

    def write(self, data):
        self.hash.update(data)
        return len(data)

    def flush(self):
        pass


def _tensor_digest(value):
    import torch

    if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
        raise HNDLError("E_UNSUPPORTED", "deployment requires dense, materialized tensors")
    # Streaming serialization avoids a dependency on NumPy, supports bfloat16,
    # and limits temporary memory to the largest individual tensor. Cloning
    # removes unrelated storage bytes behind views from the fingerprint.
    writer = _HashWriter()
    torch.save(value.detach().cpu().contiguous().clone(), writer)
    return writer.hash.hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(value):
    try:
        path = inspect.getsourcefile(value)
        return _file_digest(path) if path and os.path.isfile(path) else None
    except (TypeError, OSError):
        return None


def _semantic(value, seen=None):
    """Fingerprint Python settings, closures, and custom callable behavior."""
    import torch

    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        # repr also handles NaN/Inf settings without non-standard JSON.
        return {"float": repr(value)}
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_digest(value)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"recursive": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen = seen | {id(value)}
    if isinstance(value, Mapping):
        entries = [(_semantic(k, seen), _semantic(v, seen)) for k, v in value.items()]
        return {"mapping": sorted(entries, key=lambda item: _json(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_semantic(v, seen) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_semantic(v, seen) for v in value), key=_json)
    if isinstance(value, types.CodeType):
        return {"code": value.co_code.hex(), "constants": _semantic(value.co_consts, seen),
                "names": value.co_names, "freevars": value.co_freevars}
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__, "source": _source(value)}
    if isinstance(value, functools.partial):
        return {"partial": _semantic(value.func, seen), "args": _semantic(value.args, seen),
                "kwargs": _semantic(value.keywords, seen)}
    if inspect.ismethod(value):
        return {"method": _semantic(value.__func__, seen), "self": _semantic(value.__self__, seen)}
    if inspect.isfunction(value):
        globals_used = {name: value.__globals__[name] for name in value.__code__.co_names
                        if name in value.__globals__ and not name.startswith("__")
                        and not (value.__module__ == "hndl.operators._attention_backend"
                                 and name == "_compiled_flex")}
        return {"function": f"{value.__module__}.{value.__qualname__}",
                "code": _semantic(value.__code__, seen), "source": _source(value),
                "defaults": _semantic(value.__defaults__, seen),
                "kwdefaults": _semantic(value.__kwdefaults__, seen),
                "closure": _semantic(tuple(c.cell_contents for c in value.__closure__ or ()), seen),
                "globals": _semantic(globals_used, seen)}
    if inspect.isclass(value) or inspect.isbuiltin(value):
        return {"symbol": f"{value.__module__}.{value.__qualname__}", "source": _source(value)}
    if hasattr(value, "__dict__"):
        return {"class": _semantic(type(value), seen), "attributes": _semantic(vars(value), seen)}
    raise HNDLError("E_UNSUPPORTED", f"cannot fingerprint deployment setting of type {type(value).__qualname__}")


def _implementation(model):
    modules = []
    for name, module in model.nodes.named_modules():
        ignored = _MODULE_INTERNALS | _PREPARED_METADATA | _DERIVED_TABLES.get(type(module).__module__, set())
        attrs = {key: value for key, value in vars(module).items() if key not in ignored}
        # Forward hooks are inference behavior; serialization/backward hooks are
        # not. Include code and closure state rather than registration IDs.
        hooks = [*module._forward_pre_hooks.values(), *module._forward_hooks.values()]
        modules.append({"name": name, "class": _semantic(type(module)),
                        "forward": _semantic(type(module).forward),
                        "attributes": _semantic(attrs), "hooks": _semantic(hooks)})
    return _digest(modules)


def _state(model):
    values = []
    # Include non-persistent buffers, which state_dict() intentionally omits.
    for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        values.extend((kind, name, _tensor_digest(tensor)) for name, tensor in tensors)
    return _digest(values)


def _environment(device):
    import torch

    result = {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "hip": torch.version.hip, "torch_build": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        result["gpu"] = {"name": torch.cuda.get_device_name(device),
                         "capability": list(torch.cuda.get_device_capability(device))}
    else:
        try:
            flags = next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags"))
        except (OSError, StopIteration):
            flags = platform.processor()
        result["cpu_flags"] = hashlib.sha256(flags.encode()).hexdigest()
    return result


def _toolchain(device):
    command = shlex.split(os.environ.get("CXX") or shutil.which("g++") or "c++")
    try:
        result = subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=5)
        version = (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        version = None
    try:
        triton = metadata.version("triton")
    except metadata.PackageNotFoundError:
        triton = None
    result = {"cxx": command, "version": version, "triton": triton,
              "environment": {name: os.environ[name] for name in
                              ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CUDA_HOME", "CUDA_PATH",
                               "TORCH_CUDA_ARCH_LIST") if name in os.environ}}
    if device.type == "cuda":
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is not None:
            result["cuda_headers"] = {
                name: _file_digest(Path(CUDA_HOME) / "include" / name)
                for name in ("cuda.h", "cuda_runtime_api.h", "crt/host_defines.h", "nv/target")
                if (Path(CUDA_HOME) / "include" / name).is_file()}
    return result


def _inputs(model, example_inputs):
    import torch

    names = tuple(model.plan.inputs)
    if isinstance(example_inputs, Mapping):
        if set(example_inputs) != set(names):
            raise HNDLError("E_BINDING", f"deployment examples must provide exactly {names}")
        values = tuple(example_inputs[name] for name in names)
    elif isinstance(example_inputs, torch.Tensor):
        values = (example_inputs,)
    elif isinstance(example_inputs, (tuple, list)):
        values = tuple(example_inputs)
    else:
        raise HNDLError("E_BINDING", "deployment examples must be tensors, a tuple, or a named mapping")
    if len(values) != len(names) or not all(isinstance(v, torch.Tensor) for v in values):
        raise HNDLError("E_BINDING", f"deployment requires tensor inputs {names}")
    if any(value.requires_grad for value in values):
        raise HNDLError("E_UNSUPPORTED", "deployment is inference-only; detach the example inputs")
    if len({v.device for v in values}) != 1 or values[0].device.type not in {"cpu", "cuda"}:
        raise HNDLError("E_UNSUPPORTED", "deployment requires inputs on one CPU or CUDA device")
    return values


def _contract(name, value):
    return {"name": name, "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


@dataclass(frozen=True)
class DeploymentArtifact:
    """A retained native package and its adjacent compatibility manifest."""

    path: Path
    manifest_path: Path
    cache_key: str
    cache_hit: bool = False


def _paths(artifact):
    path = Path(artifact.path if isinstance(artifact, DeploymentArtifact) else artifact)
    return path, path.with_suffix(".json")


def inspect_artifact(artifact):
    """Read a package's manifest without importing PyTorch or loading native code."""
    _, manifest_path = _paths(artifact)
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema_version"] != _SCHEMA or _digest(manifest["specification"]) != manifest["cache_key"]:
            raise ValueError("schema or content digest mismatch")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HNDLError("E_DEPLOYMENT", f"invalid deployment manifest {manifest_path}: {error}") from error


@contextmanager
def _locked(directory):
    import fcntl

    with (directory / ".hndl-deployment.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _checked_manifest(path):
    manifest = inspect_artifact(path)
    if _file_digest(path) != manifest.get("package_sha256"):
        raise HNDLError("E_DEPLOYMENT", f"deployment package checksum mismatch: {path}")
    return manifest


def _evict(directory, keep, max_entries, max_bytes):
    entries = []
    for manifest in directory.glob("*/model.json"):
        if len(manifest.parent.name) != 64 or any(c not in "0123456789abcdef" for c in manifest.parent.name):
            continue
        package = manifest.with_suffix(".pt2")
        if package.is_file():
            entries.append((manifest.stat().st_mtime_ns, manifest.parent,
                            manifest.stat().st_size + package.stat().st_size))
    total = sum(size for _, _, size in entries)
    for _, path, size in sorted(entries):
        if len(entries) <= max_entries and total <= max_bytes:
            break
        if path == keep:
            continue
        shutil.rmtree(path)
        total -= size
        entries = [entry for entry in entries if entry[1] != path]


def export(model, example_inputs, directory, *, cache=True, max_entries=8,
           max_bytes=_DEFAULT_MAX_BYTES, inductor_configs=None):
    """Compile a fixed input signature into a cached, weight-specific .pt2 package.

    ``model`` must be an eval-mode HNDL model. The cache limits cover retained
    packages and manifests, not PyTorch's separate compiler cache. A package
    larger than ``max_bytes`` is rejected. Loading an evicted artifact requires
    exporting it again; already loaded callables remain usable.
    """
    import torch
    import torch._inductor as inductor

    if not hasattr(inductor, "aoti_compile_and_package") or not hasattr(inductor, "aoti_load_package"):
        raise HNDLError("E_UNSUPPORTED", "this PyTorch build does not provide AOTInductor packaging")
    if not hasattr(model, "plan") or not hasattr(model, "nodes"):
        raise HNDLError("E_ARGUMENT", "export expects a built HNDL model")
    if any(module.training for module in model.modules()):
        raise HNDLError("E_UNSUPPORTED", "call model.eval() before inference deployment export")
    if type(max_entries) is not int or max_entries < 1 or type(max_bytes) is not int or max_bytes < 1:
        raise HNDLError("E_ARGUMENT", "deployment cache limits must be positive integers")
    configs = {} if inductor_configs is None else dict(inductor_configs)
    try:
        _json(configs)
    except (ValueError, TypeError) as error:
        raise HNDLError("E_ARGUMENT", "inductor_configs must contain JSON-serializable values") from error
    values = _inputs(model, example_inputs)
    with torch.no_grad():
        model(*values)  # validate contracts and materialize derived lookup tables
    specification = {
        "plan": model.plan.semantic_digest, "implementation": _implementation(model),
        "state": _state(model), "environment": _environment(values[0].device),
        "toolchain": _toolchain(values[0].device),
        "inputs": [_contract(name, value) for name, value in zip(model.plan.inputs, values)],
        "outputs": list(model.plan.outputs), "inductor_configs": configs,
        "exporter": {"deployment": _file_digest(__file__),
                     "runtime": _file_digest(Path(__file__).with_name("torch.py")),
                     "prepared": (_file_digest(Path(__file__).with_name("prepared.py"))
                                  if Path(__file__).with_name("prepared.py").is_file() else None)},
    }
    module = model.prepare(values, backend="generated")._export_module if hasattr(model, "prepare") else model
    with torch.no_grad():
        program = torch.export.export(module, values, strict=False)
    # The captured graph closes a gap left by source/settings hashes: mutable
    # settings accessed through imported modules or a monkey-patched helper can
    # change execution while the operator's own source file stays unchanged.
    specification["graph"] = _digest({"code": program.graph_module.code,
                                      "signature": str(program.graph_signature),
                                      "constants": _semantic(program.constants)})
    if _state(model) != specification["state"] or _implementation(model) != specification["implementation"]:
        raise HNDLError("E_DEPLOYMENT", "model state or Python settings changed during deployment export")
    cache_key = _digest(specification)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / cache_key
    package_path, manifest_path = target / "model.pt2", target / "model.json"
    with _locked(directory):
        if cache and package_path.is_file() and manifest_path.is_file():
            try:
                manifest = _checked_manifest(package_path)
            except HNDLError:
                pass  # a damaged cache entry is rebuilt
            else:
                if manifest["specification"] == specification:
                    if package_path.stat().st_size + manifest_path.stat().st_size > max_bytes:
                        raise HNDLError("E_RESOURCE", "cached deployment package exceeds max_bytes")
                    os.utime(manifest_path, None)
                    _evict(directory, target, max_entries, max_bytes)
                    return DeploymentArtifact(package_path, manifest_path, cache_key, True)
    with tempfile.TemporaryDirectory(prefix=".hndl-build-", dir=directory) as staging_name:
        staging = Path(staging_name)
        with torch.no_grad():
            inductor.aoti_compile_and_package(program, package_path=str(staging / "model.pt2"),
                                              inductor_configs=dict(configs))
        if _state(model) != specification["state"] or _implementation(model) != specification["implementation"]:
            raise HNDLError("E_DEPLOYMENT", "model state or Python settings changed during deployment compilation")
        manifest = {"schema_version": _SCHEMA, "cache_key": cache_key, "specification": specification,
                    "package_sha256": _file_digest(staging / "model.pt2")}
        (staging / "model.json").write_text(_json(manifest) + "\n")
        size = sum(path.stat().st_size for path in staging.iterdir())
        if size > max_bytes:
            raise HNDLError("E_RESOURCE", f"deployment package needs {size} bytes, exceeding max_bytes={max_bytes}")
        with _locked(directory):
            # All package readers take the same lock, so they cannot observe a
            # package/manifest pair from different builds during replacement.
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
            _evict(directory, target, max_entries, max_bytes)
    return DeploymentArtifact(package_path, manifest_path, cache_key)


class DeploymentModel:
    """A loaded native inference callable with fixed shape/dtype/stride guards."""

    def __init__(self, runner, manifest):
        self._runner = runner
        self.manifest = manifest
        self._inputs = tuple({**contract, "shape": list(contract["shape"]), "stride": list(contract["stride"])}
                             for contract in manifest["specification"]["inputs"])
        self._names = tuple(contract["name"] for contract in self._inputs)

    def __call__(self, *inputs, **named):
        import torch

        if len(inputs) > len(self._names):
            raise HNDLError("E_BINDING", f"deployment expects inputs {self._names}")
        values = dict(zip(self._names, inputs))
        if values.keys() & named.keys():
            raise HNDLError("E_BINDING", "deployment input was supplied twice")
        values.update(named)
        if set(values) != set(self._names):
            raise HNDLError("E_BINDING", f"deployment expects inputs {self._names}")
        bound = tuple(values[name] for name in self._names)
        for value, contract in zip(bound, self._inputs):
            if not isinstance(value, torch.Tensor) or _contract(contract["name"], value) != contract:
                raise HNDLError("E_RUNTIME", f"deployment input {contract['name']!r} must match {contract}")
            if value.requires_grad:
                raise HNDLError("E_UNSUPPORTED", "deployment is inference-only; detach the inputs")
        with torch.no_grad():
            return self._runner(*bound)


def load(artifact, *, check_environment=True):
    """Load a native package after validating its checksum and runtime ABI.

    Set ``check_environment=False`` only when the caller has independently
    established compatibility of the compiler/runtime and target hardware.
    """
    import torch
    import torch._inductor as inductor

    path, _ = _paths(artifact)
    path = path.expanduser().resolve()
    cache_root = (path.parent.parent if len(path.parent.name) == 64
                  and all(c in "0123456789abcdef" for c in path.parent.name) else path.parent)
    with _locked(cache_root):
        manifest = _checked_manifest(path)
        contract = manifest["specification"]["inputs"][0]
        if check_environment and _environment(torch.device(contract["device"])) != manifest["specification"]["environment"]:
            raise HNDLError("E_DEPLOYMENT", "deployment runtime or hardware differs from the compilation environment; rebuild the package")
        if not hasattr(inductor, "aoti_load_package"):
            raise HNDLError("E_UNSUPPORTED", "this PyTorch build does not provide AOTInductor package loading")
        runner = inductor.aoti_load_package(str(path))
        os.utime(path.with_suffix(".json"), None)
    return DeploymentModel(runner, manifest)
