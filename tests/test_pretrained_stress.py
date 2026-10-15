"""Empirical stress test suite for Pretrained checkpoint caching, revision hashing,

Timm revision forwarding, training mode synchronization, and error wrapping (H-04, H-05, M-01, M-02).
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
from unittest.mock import patch

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, resolve
from hndl import pretrained as sources
from hndl.pretrained import LocalProvider, Source, TimmProvider, TransformersProvider
from hndl.torch import build, network

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("timm")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def gpt2_dir_checkpoint(tmp_path):
    """Create a minimal valid GPT-2 directory checkpoint."""
    config = transformers.GPT2Config(
        n_layer=1,
        n_embd=8,
        n_head=2,
        vocab_size=16,
        n_positions=16,
    )
    torch.manual_seed(42)
    model = transformers.GPT2LMHeadModel(config)
    path = tmp_path / "tiny-gpt2"
    model.save_pretrained(path, safe_serialization=True)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return path


@pytest.fixture
def bare_safetensors_checkpoint(gpt2_dir_checkpoint, tmp_path):
    """Separate model.safetensors and config.json into standalone paths."""
    weights_path = tmp_path / "model.safetensors"
    config_path = tmp_path / "custom_config.json"
    shutil.copyfile(gpt2_dir_checkpoint / "model.safetensors", weights_path)
    shutil.copyfile(gpt2_dir_checkpoint / "config.json", config_path)
    sources.resolve_source.cache_clear()
    sources.output_shape.cache_clear()
    return weights_path, config_path


# ==============================================================================
# 1. Checkpoint Caching & Invalidation (M-01)
# ==============================================================================


def test_resolve_source_directory_config_mutation_same_size_updates_revision(gpt2_dir_checkpoint):
    """Modifying config.json with identical byte size but different content and new mtime

    must result in a cache miss and an updated revision hash in Source.
    """
    sources.resolve_source.cache_clear()
    first_source = sources.resolve_source(str(gpt2_dir_checkpoint))

    config_file = gpt2_dir_checkpoint / "config.json"
    orig_bytes = config_file.read_bytes()
    orig_stat = config_file.stat()

    # Mutate a character without changing the length (e.g. n_embd: 8 -> n_embd: 9 or flip digits)
    if b'"n_embd": 8' in orig_bytes:
        new_bytes = orig_bytes.replace(b'"n_embd": 8', b'"n_embd": 9')
    else:
        # Fallback exact length mutation
        new_bytes = orig_bytes.replace(b'"vocab_size": 16', b'"vocab_size": 32')
    assert len(new_bytes) == len(orig_bytes), "Test setup requirement: byte size must be identical"
    assert new_bytes != orig_bytes

    config_file.write_bytes(new_bytes)
    # Ensure mtime is strictly different
    new_mtime = orig_stat.st_mtime + 2.0
    os.utime(config_file, (new_mtime, new_mtime))

    second_source = sources.resolve_source(str(gpt2_dir_checkpoint))
    assert second_source.revision != first_source.revision, (
        f"Revision must change when config.json content changes with updated mtime. "
        f"Old: {first_source.revision}, New: {second_source.revision}"
    )
    assert second_source.config != first_source.config


def test_resolve_source_directory_weights_mutation_same_size_updates_revision(gpt2_dir_checkpoint):
    """Modifying model.safetensors with identical byte size but different content and new mtime

    must result in a cache miss and an updated revision hash in Source.
    """
    sources.resolve_source.cache_clear()
    first_source = sources.resolve_source(str(gpt2_dir_checkpoint))

    weights_file = gpt2_dir_checkpoint / "model.safetensors"
    orig_bytes = bytearray(weights_file.read_bytes())
    orig_stat = weights_file.stat()

    # Invert the last byte so size is perfectly preserved but content and sha256 differ
    orig_bytes[-1] ^= 0xFF
    weights_file.write_bytes(orig_bytes)
    new_mtime = orig_stat.st_mtime + 2.0
    os.utime(weights_file, (new_mtime, new_mtime))

    second_source = sources.resolve_source(str(gpt2_dir_checkpoint))
    assert second_source.revision != first_source.revision, (
        f"Revision must change when model.safetensors is modified. "
        f"Old: {first_source.revision}, New: {second_source.revision}"
    )


def test_resolve_source_bare_safetensors_config_mutation_same_size(bare_safetensors_checkpoint):
    """Modifying config for bare safetensors with identical size and updated mtime updates revision."""
    weights_path, config_path = bare_safetensors_checkpoint
    sources.resolve_source.cache_clear()
    first_source = sources.resolve_source(str(weights_path), config_path=str(config_path))

    orig_bytes = config_path.read_bytes()
    orig_stat = config_path.stat()
    new_bytes = orig_bytes.replace(b'"vocab_size": 16', b'"vocab_size": 32')
    assert len(new_bytes) == len(orig_bytes)
    config_path.write_bytes(new_bytes)
    new_mtime = orig_stat.st_mtime + 2.0
    os.utime(config_path, (new_mtime, new_mtime))

    second_source = sources.resolve_source(str(weights_path), config_path=str(config_path))
    assert second_source.revision != first_source.revision
    assert second_source.kind == "file"


def test_resolve_source_bare_safetensors_weights_mutation_same_size(bare_safetensors_checkpoint):
    """Modifying bare safetensors weights with identical size and updated mtime updates revision."""
    weights_path, config_path = bare_safetensors_checkpoint
    sources.resolve_source.cache_clear()
    first_source = sources.resolve_source(str(weights_path), config_path=str(config_path))

    data = bytearray(weights_path.read_bytes())
    orig_stat = weights_path.stat()
    data[-1] ^= 0xAA
    weights_path.write_bytes(data)
    new_mtime = orig_stat.st_mtime + 2.0
    os.utime(weights_path, (new_mtime, new_mtime))

    second_source = sources.resolve_source(str(weights_path), config_path=str(config_path))
    assert second_source.revision != first_source.revision


def test_resolve_source_unmodified_hits_cache(gpt2_dir_checkpoint):
    """Calling resolve_source with untouched files repeatedly must hit the local cache."""
    sources.resolve_source.cache_clear()
    src1 = sources.resolve_source(str(gpt2_dir_checkpoint))
    hits_before = sources.resolve_source.cache_info().hits
    src2 = sources.resolve_source(str(gpt2_dir_checkpoint))
    hits_after = sources.resolve_source.cache_info().hits

    assert hits_after == hits_before + 1
    assert src1 is src2


# ==============================================================================
# 2. _traced_output_shapes Cache Invalidation (M-01)
# ==============================================================================


def test_traced_output_shapes_cache_invalidation_when_weights_change(gpt2_dir_checkpoint):
    """_traced_output_shapes must invalidate and recompute when checkpoint weights mutate."""
    sources.output_shape.cache_clear()
    sources.resolve_source.cache_clear()

    info0 = sources.output_shape.cache_info()
    shape1 = sources.output_shape(str(gpt2_dir_checkpoint), "", "features", "", ("B", 5))
    info1 = sources.output_shape.cache_info()
    assert info1.misses == info0.misses + 1

    # Second call without modifying checkpoint must be a cache hit
    shape2 = sources.output_shape(str(gpt2_dir_checkpoint), "", "features", "", ("B", 5))
    info2 = sources.output_shape.cache_info()
    assert info2.hits == info1.hits + 1
    assert shape1 == shape2 == ("B", 5, 8)

    # Mutate weights on disk
    weights_file = gpt2_dir_checkpoint / "model.safetensors"
    data = bytearray(weights_file.read_bytes())
    orig_stat = weights_file.stat()
    data[-1] ^= 0x55
    weights_file.write_bytes(data)
    new_mtime = orig_stat.st_mtime + 2.0
    os.utime(weights_file, (new_mtime, new_mtime))

    # Third call must miss the _traced_output_shapes cache because revision changed
    shape3 = sources.output_shape(str(gpt2_dir_checkpoint), "", "features", "", ("B", 5))
    info3 = sources.output_shape.cache_info()
    assert info3.misses == info2.misses + 1, "Cache miss must occur on traced output shapes after weights mutation"
    assert shape3 == ("B", 5, 8)


def test_traced_output_shapes_cache_invalidation_when_config_changes(gpt2_dir_checkpoint):
    """_traced_output_shapes must invalidate when config.json changes."""
    sources.output_shape.cache_clear()
    sources.resolve_source.cache_clear()

    shape1 = sources.output_shape(str(gpt2_dir_checkpoint), "", "features", "", ("B", 5))
    info1 = sources.output_shape.cache_info()

    config_file = gpt2_dir_checkpoint / "config.json"
    orig_bytes = config_file.read_bytes()
    orig_stat = config_file.stat()
    new_bytes = orig_bytes.replace(b'"vocab_size": 16', b'"vocab_size": 32')
    config_file.write_bytes(new_bytes)
    new_mtime = orig_stat.st_mtime + 2.0
    os.utime(config_file, (new_mtime, new_mtime))

    shape2 = sources.output_shape(str(gpt2_dir_checkpoint), "", "features", "", ("B", 5))
    info2 = sources.output_shape.cache_info()
    assert info2.misses == info1.misses + 1, "Cache miss must occur after config mutation"
    assert shape1 == shape2 == ("B", 5, 8)


# ==============================================================================
# 3. Training Mode Synchronization (H-04)
# ==============================================================================


def test_pretrained_training_mode_trainable_true_synchronization(gpt2_dir_checkpoint):
    """With trainable=True, wrapped model.model.training must be True immediately upon build

    and synchronize with subsequent .eval() and .train() calls.
    """
    plan = resolve(
        f'pretrained("{gpt2_dir_checkpoint}", trainable=True)',
        input_shape=("B", 5),
        output_shape=("B", 5, 8),
        input_dtype="int64",
    )
    model = build(plan, device=DEVICE)

    # Immediately after build, H-04 requires the wrapped model to be in training mode
    assert model["n0"].model.training is True, "H-04 violation: model was left in eval mode after build with trainable=True"
    assert all(param.requires_grad for param in model.parameters())

    # Verify switching to eval mode
    model.eval()
    assert model.training is False
    assert model["n0"].training is False
    assert model["n0"].model.training is False

    # Verify switching back to train mode
    model.train()
    assert model.training is True
    assert model["n0"].training is True
    assert model["n0"].model.training is True

    # Verify train(False) behaves identical to eval()
    model.train(False)
    assert model["n0"].model.training is False

    # Verify train(True) restores training mode
    model.train(True)
    assert model["n0"].model.training is True


def test_pretrained_training_mode_trainable_false_frozen(gpt2_dir_checkpoint):
    """With trainable=False (default), wrapped model.model.training must stay False

    even when model.train() is called.
    """
    plan = resolve(
        f'pretrained("{gpt2_dir_checkpoint}", trainable=False)',
        input_shape=("B", 5),
        output_shape=("B", 5, 8),
        input_dtype="int64",
    )
    model = build(plan, device=DEVICE)

    # Immediately after build, wrapped model must be in eval mode
    assert model["n0"].model.training is False
    assert not any(param.requires_grad for param in model.parameters())

    # Calling train() on frozen model must keep the wrapped model in eval mode
    model.train()
    assert model.training is True
    assert model["n0"].training is True
    assert model["n0"].model.training is False, "Frozen pretrained model must remain in eval mode"

    model.train(True)
    assert model["n0"].model.training is False

    model.eval()
    assert model["n0"].model.training is False


def test_pretrained_network_facade_training_mode_sync(gpt2_dir_checkpoint):
    """Verify that network(...) facade preserves training mode synchronization."""
    tuned = network(
        f'pretrained("{gpt2_dir_checkpoint}", trainable=True)',
        input_shape=("B", 5),
        output_shape=("B", 5, 8),
        input_dtype="int64",
        device=DEVICE,
    )
    assert tuned["n0"].model.training is True
    tuned.eval()
    assert tuned["n0"].model.training is False
    tuned.train()
    assert tuned["n0"].model.training is True

    frozen = network(
        f'pretrained("{gpt2_dir_checkpoint}", trainable=False)',
        input_shape=("B", 5),
        output_shape=("B", 5, 8),
        input_dtype="int64",
        device=DEVICE,
    )
    assert frozen["n0"].model.training is False
    frozen.train()
    assert frozen["n0"].model.training is False


# ==============================================================================
# 4. TimmProvider Revision Forwarding (H-05)
# ==============================================================================


def test_timm_provider_instantiate_passes_revision_hf_hub():
    """TimmProvider.instantiate must forward 'hf-hub:owner/repo@rev' when revision is present."""
    source = Source("hf", "timm/resnet18.a1_in1k", "commit_sha_12345", {"architecture": "resnet18"}, "", "timm")
    provider = TimmProvider(source, output="logits")

    with patch.object(provider.timm, "create_model") as mock_create:
        mock_create.return_value = nn.Linear(10, 10)
        provider.instantiate(weights=True)
        assert mock_create.call_count == 1
        model_name_arg = mock_create.call_args[0][0]
        assert model_name_arg == "hf-hub:timm/resnet18.a1_in1k@commit_sha_12345", (
            f"H-05 violation: expected revision '@commit_sha_12345' in model name, got {model_name_arg!r}"
        )
        assert mock_create.call_args[1].get("pretrained") is True


def test_timm_provider_instantiate_no_revision_hf_hub():
    """TimmProvider.instantiate must not append '@' when revision is empty."""
    source = Source("hf", "timm/resnet18.a1_in1k", "", {"architecture": "resnet18"}, "", "timm")
    provider = TimmProvider(source, output="logits")

    with patch.object(provider.timm, "create_model") as mock_create:
        mock_create.return_value = nn.Linear(10, 10)
        provider.instantiate(weights=True)
        assert mock_create.call_count == 1
        model_name_arg = mock_create.call_args[0][0]
        assert model_name_arg == "hf-hub:timm/resnet18.a1_in1k"


def test_timm_provider_instantiate_weights_false():
    """TimmProvider.instantiate(weights=False) must call create_model with architecture and pretrained=False."""
    source = Source("hf", "timm/resnet18.a1_in1k", "commit_sha", {"architecture": "resnet18"}, "", "timm")
    provider = TimmProvider(source, output="logits")

    with patch.object(provider.timm, "create_model") as mock_create:
        mock_create.return_value = nn.Linear(10, 10)
        provider.instantiate(weights=False)
        assert mock_create.call_args[0][0] == "resnet18"
        assert mock_create.call_args[1].get("pretrained") is False


def test_timm_provider_instantiate_local_directory(tmp_path):
    """TimmProvider.instantiate for directory must pass pretrained_cfg_overlay pointing to model.safetensors."""
    source = Source("directory", str(tmp_path), "rev_dir", {"architecture": "resnet18"}, str(tmp_path / "config.json"), "timm")
    provider = TimmProvider(source, output="logits")

    with patch.object(provider.timm, "create_model") as mock_create:
        mock_create.return_value = nn.Linear(10, 10)
        provider.instantiate(weights=True)
        assert mock_create.call_args[0][0] == "resnet18"
        assert mock_create.call_args[1].get("pretrained") is True
        overlay = mock_create.call_args[1].get("pretrained_cfg_overlay", {})
        assert overlay.get("file") == str(tmp_path / "model.safetensors")


# ==============================================================================
# 5. Error Handling & Structured Exceptions (M-02)
# ==============================================================================


def test_error_handling_corrupted_config_json(tmp_path):
    """Corrupted JSON in config.json must raise HNDLError('E_PRETRAINED', ...)."""
    bad_dir = tmp_path / "corrupted_config_dir"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("{ unquoted_key: missing_bracket", encoding="utf-8")

    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(bad_dir))
    assert exc_info.value.code == "E_PRETRAINED"
    assert "Cannot read config" in str(exc_info.value)


def test_error_handling_invalid_utf8_config_json(tmp_path):
    """Invalid UTF-8 bytes in config.json must raise HNDLError('E_PRETRAINED', ...)."""
    bad_dir = tmp_path / "invalid_utf8_dir"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_bytes(b"\xfe\xff\xff\xfe")

    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(bad_dir))
    assert exc_info.value.code == "E_PRETRAINED"
    assert "Cannot read config" in str(exc_info.value)


def test_error_handling_truncated_safetensors_file(tmp_path):
    """Truncated or corrupted .safetensors weights must raise HNDLError('E_PRETRAINED', ...)."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model_type": "gpt2"}))
    weights_file = tmp_path / "corrupted.safetensors"
    # Write 16 invalid bytes (too small to form a valid safetensors header)
    weights_file.write_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08")

    source = Source("file", str(weights_file), "rev", {"model_type": "gpt2"}, str(config_file), "transformers")
    provider = TransformersProvider(source, output="features", component="")

    with pytest.raises(HNDLError) as exc_info:
        provider.instantiate(weights=True)
    assert exc_info.value.code == "E_PRETRAINED"
    assert "Cannot load weights from" in str(exc_info.value)


def test_error_handling_timm_truncated_safetensors(tmp_path):
    """TimmProvider loading truncated bare safetensors must raise HNDLError('E_PRETRAINED', ...)."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"architecture": "resnet18"}))
    weights_file = tmp_path / "corrupted.safetensors"
    weights_file.write_bytes(b"bad_safetensors_bytes_here")

    source = Source("file", str(weights_file), "rev", {"architecture": "resnet18"}, str(config_file), "timm")
    provider = TimmProvider(source, output="logits")

    with pytest.raises(HNDLError) as exc_info:
        provider.instantiate(weights=True)
    assert exc_info.value.code == "E_PRETRAINED"
    assert "Cannot load weights from" in str(exc_info.value)


def test_error_handling_nonexistent_files(tmp_path):
    """Non-existent directory, bare safetensors, or config must raise HNDLError('E_PRETRAINED', ...)."""
    nonexistent = tmp_path / "does_not_exist"

    # Non-existent directory / file
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(nonexistent))
    assert exc_info.value.code == "E_PRETRAINED"

    # Existing safetensors, non-existent config
    fake_sf = tmp_path / "dummy.safetensors"
    fake_sf.write_bytes(b"dummy")
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(fake_sf), config_path=str(tmp_path / "missing_config.json"))
    assert exc_info.value.code == "E_PRETRAINED"
    assert "does not exist" in str(exc_info.value)


def test_error_handling_missing_safetensors_extra(monkeypatch):
    """When safetensors is unavailable, _require('safetensors') and instantiate must fail with E_PRETRAINED."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "safetensors":
            raise ImportError("Simulated missing safetensors library")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(HNDLError) as exc_info:
        sources._require("safetensors")
    assert exc_info.value.code == "E_PRETRAINED"
    assert "pip install 'hndl[pretrained]'" in str(exc_info.value)
    assert "safetensors missing" in str(exc_info.value)

    # Verify TimmProvider.instantiate triggers the guard
    source = Source("hf", "timm/resnet18", "rev", {"architecture": "resnet18"}, "", "timm")
    provider = TimmProvider(source, output="logits")
    with pytest.raises(HNDLError) as exc_info:
        provider.instantiate(weights=True)
    assert exc_info.value.code == "E_PRETRAINED"
    assert "safetensors missing" in str(exc_info.value)


def test_error_handling_state_dict_validation(tmp_path):
    """Empirically test .pth state dict error wrapping (nonexistent, corrupted, mismatch, non-dict)."""
    # 1. Nonexistent .pth
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(tmp_path / "missing.pth"), provider="test_prov", checksum="a" * 64)
    assert exc_info.value.code == "E_PRETRAINED"
    assert "does not exist" in str(exc_info.value)

    # 2. Checksum length validation
    real_pth = tmp_path / "weights.pth"
    real_pth.write_bytes(b"dummy_weights")
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(real_pth), provider="test_prov", checksum="short")
    assert exc_info.value.code == "E_PRETRAINED"
    assert "sha256 must be 64 lowercase hex characters" in str(exc_info.value)

    # 3. Checksum mismatch
    actual_hash = hashlib.sha256(b"dummy_weights").hexdigest()
    with pytest.raises(HNDLError) as exc_info:
        sources.resolve_source(str(real_pth), provider="test_prov", checksum="0" * 64)
    assert exc_info.value.code == "E_PRETRAINED"
    assert "not the pinned" in str(exc_info.value)

    # Register dummy provider for instantiation tests
    reg = Registry()
    reg.pretrained_provider("test_prov", lambda: nn.Linear(4, 2))

    # 4. Corrupted .pth file (fails torch.load)
    with reg.activated():
        src = Source("state_dict", str(real_pth), actual_hash[:16], {"provider": "test_prov", "sha256": actual_hash}, "", "local")
        local_prov = LocalProvider(src, output="features", component="", layer="")
        with pytest.raises(HNDLError) as exc_info:
            local_prov.instantiate(weights=True)
        assert exc_info.value.code == "E_PRETRAINED"
        assert "Cannot load weights from" in str(exc_info.value)

    # 5. .pth holds a non-dict object (e.g. list)
    list_pth = tmp_path / "list_weights.pth"
    torch.save([1, 2, 3], list_pth)
    list_hash = hashlib.sha256(list_pth.read_bytes()).hexdigest()
    with reg.activated():
        src_list = Source("state_dict", str(list_pth), list_hash[:16], {"provider": "test_prov", "sha256": list_hash}, "", "local")
        prov_list = LocalProvider(src_list, output="features", component="", layer="")
        with pytest.raises(HNDLError) as exc_info:
            prov_list.instantiate(weights=True)
        assert exc_info.value.code == "E_PRETRAINED"
        assert "holds list, not a state_dict" in str(exc_info.value)
