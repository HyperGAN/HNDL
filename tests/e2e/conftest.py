"""Pytest fixtures and configuration for HNDL E2E test suite."""

import pytest
import torch

from hndl import pretrained as sources


@pytest.fixture
def device():
    """Default device for E2E tests."""
    return "cpu"


@pytest.fixture
def text_checkpoint(tmp_path):
    """Local minimal GPT2 checkpoint with config.json and model.safetensors."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

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

    if hasattr(sources.resolve_source, "cache_clear"):
        sources.resolve_source.cache_clear()
    if hasattr(sources.output_shape, "cache_clear"):
        sources.output_shape.cache_clear()

    return path


@pytest.fixture
def vision_checkpoint(tmp_path):
    """Local minimal ViT checkpoint with config.json and model.safetensors."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

    config = transformers.ViTConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=8,
        patch_size=4,
        num_labels=3,
    )
    torch.manual_seed(42)
    model = transformers.ViTForImageClassification(config)
    path = tmp_path / "tiny-vit"
    model.save_pretrained(path, safe_serialization=True)

    if hasattr(sources.resolve_source, "cache_clear"):
        sources.resolve_source.cache_clear()
    if hasattr(sources.output_shape, "cache_clear"):
        sources.output_shape.cache_clear()

    return path
