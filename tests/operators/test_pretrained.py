"""Operator declaration, provider dispatch, error handling, and extra_repr for ``pretrained``."""

import hashlib

import pytest
import torch
from torch import nn

from hndl import HNDLError, Registry, resolve
from hndl.torch import build


def test_pretrained_declaration_attributes():
    reg = Registry.builtins()
    spec = reg.get("pretrained")
    assert spec.alias == "pretrained"
    assert spec.category == "pretrained"
    assert spec.reference is None, "pretrained has no synthetic reference oracle"
    assert "source" in spec.args
    assert "output" in spec.args


def test_pretrained_unknown_provider_raises_e_pretrained(tmp_path):
    reg = Registry.builtins()
    path = tmp_path / "mock.pt"
    path.touch()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(HNDLError, match="E_PRETRAINED"):
        resolve(f'pretrained("{path}", provider="nonexistent_provider", sha256="{digest}")',
                input_shape=("B", 10), output_shape=("B", 5), registry=reg)


def test_pretrained_missing_sha256_raises_e_pretrained(tmp_path):
    reg = Registry.builtins()
    reg.pretrained_provider("mock_mlp", lambda: nn.Sequential(nn.Linear(10, 5)))
    path = tmp_path / "mock.pt"
    path.touch()
    with pytest.raises(HNDLError, match="E_PRETRAINED"):
        resolve(f'pretrained("{path}", provider="mock_mlp")',
                input_shape=("B", 10), output_shape=("B", 5), registry=reg)


def test_pretrained_custom_trusted_provider(tmp_path):
    reg = Registry.builtins()
    reg.pretrained_provider("mock_mlp", lambda: nn.Sequential(nn.Linear(10, 5)))
    path = tmp_path / "mock.pt"
    torch.save(nn.Sequential(nn.Linear(10, 5)).state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = resolve(f'pretrained("{path}", provider="mock_mlp", sha256="{digest}")',
                   input_shape=("B", 10), output_shape=("B", 5), registry=reg)
    model = build(plan, device="cpu")
    x = torch.randn(2, 10)
    out = model(x=x)["output"]
    assert out.shape == (2, 5)


def test_pretrained_trainable_mode_sync(tmp_path):
    reg = Registry.builtins()
    reg.pretrained_provider("mock_trainable", lambda: nn.Linear(10, 5))
    path = tmp_path / "mock_trainable.pt"
    torch.save(nn.Linear(10, 5).state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = resolve(f'pretrained("{path}", provider="mock_trainable", sha256="{digest}", trainable=True)',
                   input_shape=("B", 10), output_shape=("B", 5), registry=reg)
    model = build(plan, device="cpu")
    assert model.training is True
    assert model["n0"].model.training is True
    model.eval()
    assert model["n0"].model.training is False
