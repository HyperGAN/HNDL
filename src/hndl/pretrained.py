"""Checkpoint sources and architecture providers behind the ``pretrained`` operator.

A source is a local directory holding ``config.json`` and safetensors weights,
a single ``.safetensors`` file plus a ``config.json`` path, ``hf://repo[@revision]``,
or a local ``.pth`` state dict named together with a registered provider. The
architecture comes from the checkpoint's own configuration through
``transformers`` or ``timm``, or — for a ``.pth`` file, which describes no
architecture at all — from a builder the host registered on its ``Registry``.
HNDL never defines network layouts here, and nothing in a checkpoint is
executed as code: ``.pth`` files are read with ``weights_only=True`` after
their ``sha256`` has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from .errors import HNDLError
from .types import Immutable

OUTPUTS = ("features", "pooled", "logits", "embeds")
_HF_PREFIX = "hf://"
_STATE_DICT_SUFFIXES = (".pth", ".pt")
_HEX = "0123456789abcdef"
_MAX_LISTED_MODULES = 40


def _fail(message):
    raise HNDLError("E_PRETRAINED", message)


def _require(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        _fail(f"The pretrained operator needs the optional extra: pip install 'hndl[pretrained]' ({module_name} missing)")


@dataclass(frozen=True)
class Source(Immutable):
    """A resolved checkpoint location."""

    kind: str            # "hf", "directory", "file", or "state_dict"
    location: str        # repo id, directory, safetensors file, or .pth file path
    revision: str        # commit hash for hf; content digest otherwise
    config: dict         # parsed config.json, or {"provider", "sha256"} for a .pth file
    config_path: str     # where config.json was read from ("" for a .pth file)
    provider: str        # "transformers", "timm", or "local"

    @property
    def spec(self):
        return f"{_HF_PREFIX}{self.location}@{self.revision}" if self.kind == "hf" else self.location


def _digest(*parts):
    hasher = sha256()
    for part in parts:
        hasher.update(part.encode("utf-8") if isinstance(part, str) else part)
    return hasher.hexdigest()[:16]


_file_digests = {}


def file_sha256(path):
    """The full hex digest of a file, memoized while its size and mtime hold."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    digest = _file_digests.get(key)
    if digest is None:
        hasher = sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        digest = _file_digests[key] = hasher.hexdigest()
    return digest


def _verify_digest(path, expected):
    """Hash the file and compare it with the pinned digest, failing loudly."""
    actual = file_sha256(path)
    if actual != expected:
        _fail(f"{path}: sha256 is {actual}, not the pinned {expected}; the checkpoint file changed")
    return actual


def _provider_for(config):
    if "model_type" in config:
        return "transformers"
    if "architecture" in config:
        return "timm"
    _fail("config.json declares neither a transformers model_type nor a timm architecture")


@lru_cache(maxsize=64)
def resolve_source(source, config_path="", provider="", checksum=""):
    """Locate a checkpoint and read its configuration without loading weights."""
    if type(source) is not str or not source:
        _fail("source must be a non-empty string: a local path or hf://repo[@revision]")
    if provider or Path(source).expanduser().suffix in _STATE_DICT_SUFFIXES and not source.startswith(_HF_PREFIX):
        return _resolve_state_dict(source, config_path, provider, checksum)
    if checksum:
        _fail("sha256= pins a local .pth checkpoint; transformers and timm checkpoints record a revision instead")
    if source.startswith(_HF_PREFIX):
        if config_path:
            _fail("config= applies to local checkpoints; hf:// repositories carry their own config.json")
        repo, _, revision = source[len(_HF_PREFIX):].partition("@")
        if not repo or repo.count("/") != 1:
            _fail(f"hf:// sources are written hf://owner/repo[@revision], got {source!r}")
        hub = _require("huggingface_hub")
        try:
            path = hub.hf_hub_download(repo, "config.json", revision=revision or None)
        except Exception as exc:  # network, auth, missing repo
            _fail(f"Cannot fetch config.json for {repo!r}: {type(exc).__name__}: {exc}")
        # Cache layout: .../snapshots/<commit>/config.json (a symlink into blobs/).
        snapshot = Path(path)
        commit = snapshot.parent.name if snapshot.parent.parent.name == "snapshots" else _digest(snapshot.read_bytes())
        config = json.loads(snapshot.read_text(encoding="utf-8"))
        return Source("hf", repo, commit, config, str(snapshot), _provider_for(config))
    path = Path(source).expanduser()
    if path.is_dir():
        if config_path:
            _fail("config= applies to a bare .safetensors file; a checkpoint directory holds its own config.json")
        config_file = path / "config.json"
        if not config_file.is_file():
            _fail(f"{path} has no config.json; pass a bare .safetensors file with config=")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        return Source("directory", str(path), _digest(config_file.read_bytes()), config, str(config_file), _provider_for(config))
    if path.is_file() and path.suffix == ".safetensors":
        if not config_path:
            _fail("A bare .safetensors file needs config=<path to config.json>; tensors alone do not describe an architecture")
        config_file = Path(config_path).expanduser()
        if not config_file.is_file():
            _fail(f"config file {config_file} does not exist")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        revision = _digest(config_file.read_bytes(), str(path.stat().st_size))
        return Source("file", str(path), revision, config, str(config_file), _provider_for(config))
    _fail(f"{source!r} is not an hf:// repository, a checkpoint directory, or a .safetensors file")


def _resolve_state_dict(source, config_path, provider, checksum):
    """Pin a local ``.pth`` state dict to a registered architecture provider."""
    if config_path:
        _fail("config= applies to transformers and timm checkpoints; a .pth file names its architecture with provider=")
    if not provider:
        _fail(f"{source!r} is a bare state dict and describes no architecture; "
              "pass provider=<name registered with registry.pretrained_provider(...)>")
    path = Path(source).expanduser()
    if path.suffix not in _STATE_DICT_SUFFIXES:
        _fail(f"provider= loads a local state dict; {source!r} does not end in {' or '.join(_STATE_DICT_SUFFIXES)}")
    if not path.is_file():
        _fail(f"{path} does not exist")
    if not checksum:
        _fail(f"{path} needs sha256=<64 hex characters>; a local checkpoint is only loaded when its digest is pinned "
              f"(its digest today is {file_sha256(path)})")
    if len(checksum) != 64 or any(character not in _HEX for character in checksum):
        _fail(f"sha256 must be 64 lowercase hex characters, got {checksum!r}")
    # Resolution verifies the pinned digest, so a restored plan whose file changed fails here.
    digest = _verify_digest(path, checksum)
    provider_build(provider)
    return Source("state_dict", str(path), digest[:16], {"provider": provider, "sha256": digest}, "", "local")


def unverified_state_dict_source(source, provider, checksum, revision):
    """Rebuild a ``.pth`` source from concrete plan arguments without touching the file.

    The allocation-free meta probe needs only the architecture; the digest was
    verified when the plan resolved and is verified again before any weights
    are read, so the probe never hashes or opens the checkpoint.
    """
    provider_build(provider)
    return Source("state_dict", str(Path(source).expanduser()), revision,
                  {"provider": provider, "sha256": checksum}, "", "local")


def provider_build(name):
    """The trusted builder registered for ``name`` on the active registry."""
    from .registry import Registry
    registry = Registry.active()
    build = registry.pretrained_builder(name) if registry is not None else None
    if build is None:
        known = ", ".join(registry.pretrained_providers) if registry is not None else ""
        _fail(f"No pretrained provider named {name!r} is registered{' (known: ' + known + ')' if known else ''}; "
              "register one with registry.pretrained_provider(name, build) and pass that registry to resolve/build")
    return build


def provider_readout(provider, name):
    """The trusted readout callable the active registry binds to ``provider``."""
    from .registry import Registry
    registry = Registry.active()
    readouts = registry.pretrained_readouts(provider) if registry is not None else {}
    readout = readouts.get(name)
    if readout is None:
        known = ", ".join(sorted(readouts)) or "(none)"
        _fail(f"Pretrained provider {provider!r} has no readout named {name!r} (registered: {known}); "
              "register one with registry.pretrained_provider(name, build, readouts={\"<name>\": fn}) "
              "or registry.pretrained_readout(provider, name, fn)")
    return readout


@dataclass(frozen=True)
class Contract(Immutable):
    """What the wrapped model consumes."""

    kind: str                 # "ids", "pixels", or "tensor" (a provider model, contract unknown)
    channels: object = None   # for pixels
    image_size: object = None  # (H, W) for pixels when the architecture fixes it


class TransformersProvider(Immutable):
    name = "transformers"

    def __init__(self, source, output, component):
        self.source, self.output, self.component = source, output, component
        transformers = _require("transformers")
        self.transformers = transformers
        self.config = transformers.AutoConfig.from_pretrained(source.config_path)
        towers = [name for name in ("vision", "text") if hasattr(self.config, f"{name}_config")]
        if towers and not component:
            _fail(f"{source.location} has {' and '.join(towers)} towers; pass component=\"vision\" or component=\"text\"")
        if component and component not in towers:
            _fail(f"component must be one of {towers or ['(none)']} for {source.location}, got {component!r}")
        self.tower = getattr(self.config, f"{component}_config") if component else self.config
        if output not in OUTPUTS and not output.isidentifier():
            _fail(f"output must be one of {', '.join(OUTPUTS)} or an output attribute name, got {output!r}")

    def contract(self):
        image_size = getattr(self.tower, "image_size", None)
        channels = getattr(self.tower, "num_channels", 3)
        if image_size is not None and getattr(self.tower, "vocab_size", None) is None or self.component == "vision":
            size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
            return Contract("pixels", channels, tuple(size))
        return Contract("ids")

    def _class(self):
        transformers = self.transformers
        if self.component:
            prefix = type(self.config).__name__.removesuffix("Config")
            suffix = "WithProjection" if self.output == "embeds" else ""
            name = f"{prefix}{self.component.capitalize()}Model{suffix}"
            cls = getattr(transformers, name, None)
            if cls is None:
                _fail(f"transformers has no {name}; output={self.output!r} is unavailable for this component")
            return cls, None
        if self.output == "logits":
            for name in ("AutoModelForCausalLM", "AutoModelForImageClassification", "AutoModelForSequenceClassification",
                         "AutoModelForMaskedLM"):
                auto = getattr(transformers, name)
                try:
                    if type(self.config) in auto._model_mapping:
                        return auto, name
                except Exception:
                    continue
            _fail(f"No transformers head with logits is registered for model_type {self.config.model_type!r}")
        if self.output == "embeds":
            _fail("output=\"embeds\" needs a projection model; use component=\"vision\" or \"text\" on a CLIP-style checkpoint")
        return transformers.AutoModel, "AutoModel"

    def _from_config(self, cls):
        config = self.tower if self.component else self.config
        # Auto classes dispatch with from_config; concrete tower classes take the config directly.
        return cls.from_config(config) if hasattr(cls, "from_config") else cls(config)

    def instantiate(self, weights):
        cls, auto_name = self._class()
        if not weights:
            return self._from_config(cls)
        location = self.source.location if self.source.kind != "file" else None
        if self.source.kind == "file":
            model = self._from_config(cls)
            self._load_safetensors(model)
            return model
        kwargs = {"revision": self.source.revision} if self.source.kind == "hf" else {}
        # transformers refuses to load under a torch.device context without
        # accelerate; load on the CPU and let the backend move the module.
        with torch.device("cpu"):
            return cls.from_pretrained(location, **kwargs)

    def _load_safetensors(self, model):
        _require("safetensors")
        from safetensors.torch import load_file
        state = load_file(self.source.location)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Tied parameters (e.g. GPT-2's lm_head) are not stored; retie them.
        tied = getattr(model, "_tied_weights_keys", None) or ()
        tied = set(tied.keys() if isinstance(tied, dict) else tied)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        missing = [key for key in missing if key not in tied]
        if missing or unexpected:
            _fail(f"{self.source.location}: missing keys {sorted(missing)[:5]}, unexpected keys {sorted(unexpected)[:5]}")

    def select(self, result):
        embeds = {"vision": "image_embeds", "text": "text_embeds"}.get(self.component, "embeds")
        attribute = {"features": "last_hidden_state", "pooled": "pooler_output", "logits": "logits",
                     "embeds": embeds}.get(self.output, self.output)
        value = getattr(result, attribute, None) if not isinstance(result, dict) else result.get(attribute)
        if value is None:
            available = [key for key in (result.keys() if hasattr(result, "keys") else []) if getattr(result, key, None) is not None]
            _fail(f"Output {self.output!r} ({attribute}) is not produced by this model; available: {available}")
        return value

    def call(self, model, x, kind):
        return model(**{"input_ids" if kind == "ids" else "pixel_values": x})


class TimmProvider(Immutable):
    name = "timm"

    def __init__(self, source, output, component):
        self.source, self.output = source, output
        if component:
            _fail("component= applies to multi-tower transformers checkpoints, not timm models")
        if output not in ("features", "pooled", "logits"):
            _fail(f"timm checkpoints support output features, pooled, or logits, got {output!r}")
        self.timm = _require("timm")
        self.architecture = source.config["architecture"]
        pretrained_cfg = source.config.get("pretrained_cfg", {})
        self.input_size = tuple(pretrained_cfg.get("input_size", (3, 224, 224)))
        self.num_classes = source.config.get("num_classes", pretrained_cfg.get("num_classes"))

    def contract(self):
        return Contract("pixels", self.input_size[0], tuple(self.input_size[1:]))

    def instantiate(self, weights):
        kwargs = {} if self.num_classes is None else {"num_classes": self.num_classes}
        if not weights or self.source.kind == "file":
            model = self.timm.create_model(self.architecture, pretrained=False, **kwargs)
            if weights:
                from safetensors.torch import load_file
                missing, unexpected = model.load_state_dict(load_file(self.source.location), strict=False)
                if missing or unexpected:
                    _fail(f"{self.source.location}: missing keys {sorted(missing)[:5]}, unexpected keys {sorted(unexpected)[:5]}")
            return model
        with torch.device("cpu"):
            if self.source.kind == "hf":
                return self.timm.create_model(f"hf-hub:{self.source.location}", pretrained=True, **kwargs)
            return self.timm.create_model(self.architecture, pretrained=True, **kwargs,
                                          pretrained_cfg_overlay={"file": str(Path(self.source.location) / "model.safetensors")})

    def select(self, result):
        return result

    def call(self, model, x, kind):
        if self.output == "logits":
            return model(x)
        features = model.forward_features(x)
        if self.output == "features":
            return features
        return model.forward_head(features, pre_logits=True)


class _Captured(BaseException):
    """Unwinds a forward pass once the requested submodule has produced its output."""

    def __init__(self, value):
        super().__init__()
        self.value = value


def submodule(model, name, location):
    """The submodule at a dotted ``named_modules()`` path, or an ``E_PRETRAINED`` failure."""
    try:
        return model.get_submodule(name)
    except AttributeError:
        available = [candidate for candidate, _ in model.named_modules() if candidate]
        listed = ", ".join(available[:_MAX_LISTED_MODULES]) or "(none)"
        more = f", ... ({len(available)} submodules)" if len(available) > _MAX_LISTED_MODULES else ""
        _fail(f"{location} has no submodule {name!r}; available: {listed}{more}")


def capture_layer(model, name, x, location):
    """Run ``model(x)`` and return the named submodule's output, stopping there."""
    target = submodule(model, name, location)

    def hook(module, inputs, output):
        raise _Captured(output)

    handle = target.register_forward_hook(hook)
    try:
        model(x)
    except _Captured as captured:
        value = captured.value
    else:
        _fail(f"{location}: submodule {name!r} did not run during the forward pass")
    finally:
        handle.remove()
    if not isinstance(value, torch.Tensor):
        _fail(f"{location}: submodule {name!r} returns {type(value).__name__}, not a tensor")
    return value


def call_readout(model, readout, name, x, location):
    """Run a host-registered readout and require exactly one tensor back."""
    value = readout(model, x)
    if not isinstance(value, torch.Tensor):
        _fail(f"{location}: readout {name!r} returned {type(value).__name__}, not a tensor; a readout must return "
              "one tensor — combine several with torch.cat or torch.stack inside the readout, or register one "
              "readout per tensor you need")
    return value


class LocalProvider:
    """A ``.pth`` state dict loaded into an architecture the host registered."""

    name = "local"

    def __init__(self, source, output, component, layer, readout=""):
        self.source, self.layer, self.readout = source, layer, readout
        if component:
            _fail("component= applies to multi-tower transformers checkpoints, not provider checkpoints")
        if layer and readout:
            _fail(f"layer={layer!r} and readout={readout!r} both say what the node returns; pass one of them")
        if output != "features":
            _fail(f'provider checkpoints take an intermediate tensor with layer="<dotted submodule path>" '
                  f'or readout="<registered readout>"; output={output!r} applies to transformers and timm checkpoints')
        self.build = provider_build(source.config["provider"])
        self.read = provider_readout(source.config["provider"], readout) if readout else None

    def contract(self):
        return Contract("tensor")

    def instantiate(self, weights):
        model = self.build()
        if not isinstance(model, torch.nn.Module):
            _fail(f"Pretrained provider {self.source.config['provider']!r} returned "
                  f"{type(model).__name__}, not an nn.Module")
        if weights:
            path = Path(self.source.location)
            _verify_digest(path, self.source.config["sha256"])
            # weights_only=True: tensors and plain containers, never arbitrary pickled objects.
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                _fail(f"{path} holds {type(state).__name__}, not a state_dict mapping names to tensors")
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                _fail(f"{path} does not fit provider {self.source.config['provider']!r}: {exc}")
        if self.layer:
            submodule(model, self.layer, self.source.location)
        return model

    def select(self, result):
        return result

    def call(self, model, x, kind):
        if self.readout:
            return call_readout(model, self.read, self.readout, x, self.source.location)
        if not self.layer:
            return model(x)
        return capture_layer(model, self.layer, x, self.source.location)


def provider_for(source, output, component, layer="", readout=""):
    if source.provider == "local":
        return LocalProvider(source, output, component, layer, readout)
    if layer:
        _fail("layer= names a submodule of a provider checkpoint; transformers and timm checkpoints select output=")
    if readout:
        _fail("readout= names a readout the host registered with a provider checkpoint's builder; "
              "transformers and timm checkpoints select output=")
    if source.provider == "timm":
        return TimmProvider(source, output, component)
    return TransformersProvider(source, output, component)


def output_shape(source_key, config_path, output, component, input_shape, provider="", checksum="", layer="",
                 readout=""):
    """Trace the wrapped model on the meta device to learn its output shape."""
    # The builder and readout are part of the cache key: two registries may bind the same names.
    build = provider_build(provider) if provider else None
    read = provider_readout(provider, readout) if provider and readout else None
    return _traced_output_shape(source_key, config_path, output, component, input_shape, provider, checksum, layer,
                                readout, build, read)


@lru_cache(maxsize=256)
def _traced_output_shape(source_key, config_path, output, component, input_shape, provider_name, checksum, layer,
                         readout, build, read):
    source = resolve_source(source_key, config_path, provider_name, checksum)
    provider = provider_for(source, output, component, layer, readout)
    kind = provider.contract().kind
    with torch.device("meta"):
        model = provider.instantiate(weights=False)
        model.eval()
        dtype = torch.long if kind == "ids" else torch.float32
        probe = torch.zeros((2, *input_shape[1:]), dtype=dtype)
        try:
            result = provider.select(provider.call(model, probe, kind))
        except HNDLError:
            raise
        except Exception as exc:
            through = f" through readout {readout!r}" if readout else ""
            _fail(f"{source.location}: the checkpoint's model cannot consume input shape {list(input_shape)}{through}: "
                  f"{type(exc).__name__}: {exc}")
    if not isinstance(result, torch.Tensor):
        _fail(f"{source.location}: output {output!r} is not a tensor")
    return ("B", *tuple(int(d) for d in result.shape[1:]))


output_shape.cache_clear = _traced_output_shape.cache_clear
output_shape.cache_info = _traced_output_shape.cache_info


def is_offline():
    return os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")
