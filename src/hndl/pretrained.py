"""Checkpoint sources and architecture providers behind the ``pretrained`` operator.

A source is a local directory holding ``config.json`` and safetensors weights,
a single ``.safetensors`` file plus a ``config.json`` path, or ``hf://repo[@revision]``.
The architecture comes from the checkpoint's own configuration through
``transformers`` or ``timm``; HNDL never defines network layouts here. Nothing
in a checkpoint is executed as code.
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

    kind: str            # "hf", "directory", or "file"
    location: str        # repo id, directory, or safetensors file path
    revision: str        # commit hash for hf; content digest otherwise
    config: dict         # parsed config.json
    config_path: str     # where config.json was read from
    provider: str        # "transformers" or "timm"

    @property
    def spec(self):
        return f"{_HF_PREFIX}{self.location}@{self.revision}" if self.kind == "hf" else self.location


def _digest(*parts):
    hasher = sha256()
    for part in parts:
        hasher.update(part.encode("utf-8") if isinstance(part, str) else part)
    return hasher.hexdigest()[:16]


def _provider_for(config):
    if "model_type" in config:
        return "transformers"
    if "architecture" in config:
        return "timm"
    _fail("config.json declares neither a transformers model_type nor a timm architecture")


@lru_cache(maxsize=64)
def resolve_source(source, config_path=""):
    """Locate a checkpoint and read its configuration without loading weights."""
    if type(source) is not str or not source:
        _fail("source must be a non-empty string: a local path or hf://repo[@revision]")
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


@dataclass(frozen=True)
class Contract(Immutable):
    """What the wrapped model consumes."""

    kind: str                 # "ids" or "pixels"
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


def provider_for(source, output, component):
    if source.provider == "timm":
        return TimmProvider(source, output, component)
    return TransformersProvider(source, output, component)


@lru_cache(maxsize=256)
def output_shape(source_key, config_path, output, component, input_shape):
    """Trace the wrapped model on the meta device to learn its output shape."""
    source = resolve_source(source_key, config_path)
    provider = provider_for(source, output, component)
    kind = provider.contract().kind
    with torch.device("meta"):
        model = provider.instantiate(weights=False)
        model.eval()
        dtype = torch.long if kind == "ids" else torch.float32
        probe = torch.zeros((2, *input_shape[1:]), dtype=dtype)
        try:
            result = provider.select(provider.call(model, probe, kind))
        except Exception as exc:
            _fail(f"{source.location}: the checkpoint's model cannot consume input shape {list(input_shape)}: {type(exc).__name__}: {exc}")
    if not isinstance(result, torch.Tensor):
        _fail(f"{source.location}: output {output!r} is not a tensor")
    return ("B", *tuple(int(d) for d in result.shape[1:]))


def is_offline():
    return os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")
