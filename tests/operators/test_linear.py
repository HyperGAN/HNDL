"""Spectral normalization on ``linear``: parity, plumbing, and train/eval behavior."""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrizations

from hndl import resolve
from hndl.errors import HNDLError
from hndl.torch import build

ORIGINAL = "parametrizations.weight.original"


def spectral_linear(out_features, in_features, bias=True):
    """A reference layer built with torch's own parametrization API."""
    return parametrizations.spectral_norm(nn.Linear(in_features, out_features, bias=bias))


def largest_singular_value(weight):
    return torch.linalg.matrix_norm(weight.detach().reshape(weight.shape[0], -1), 2).item()


def test_registered_state_is_renamed_by_the_parametrization():
    plan = resolve("linear(8, spectral_norm=True)", input_shape=("B", 6), output_shape=("B", 8))
    assert plan.nodes[0].args["spectral_norm"] is True
    model = build(plan, device="cpu", initialization_seed=3)
    module = model[plan.nodes[0].id]
    assert sorted(name for name, _ in module.named_parameters()) == ["bias", ORIGINAL]
    assert sorted(name for name, _ in module.named_buffers()) == ["parametrizations.weight.0._u",
                                                                  "parametrizations.weight.0._v"]
    assert module.weight.shape == (8, 6)
    assert model.build_receipt["state_bytes"] == sum(
        t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))


def test_disabled_spectral_norm_leaves_the_layer_untouched(device):
    plan = resolve("linear(8)", input_shape=("B", 6), output_shape=("B", 8))
    assert plan.nodes[0].args["spectral_norm"] is False
    model = build(plan, device=device, initialization_seed=5)
    module = model[plan.nodes[0].id]
    assert sorted(name for name, _ in module.named_parameters()) == ["bias", "weight"]
    assert not list(module.named_buffers())
    reference = nn.Linear(6, 8).to(device)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(4, 6, device=device)
    torch.testing.assert_close(model(x=x)["output"], reference(x))


def test_forward_and_gradients_match_torch_spectral_norm(device):
    plan = resolve("linear(8, spectral_norm=True)", input_shape=("B", 6), output_shape=("B", 8))
    model = build(plan, device=device, initialization_seed=7)
    module = model[plan.nodes[0].id]
    reference = spectral_linear(8, 6).to(device)
    reference.load_state_dict(module.state_dict())

    x = torch.randn(4, 6, device=device)
    actual_input = x.clone().requires_grad_()
    expected_input = x.clone().requires_grad_()
    actual = model(x=actual_input)["output"]
    expected = reference(expected_input)
    torch.testing.assert_close(actual, expected)

    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(module.parametrizations.weight.original.grad,
                               reference.parametrizations.weight.original.grad)


def test_effective_weight_has_unit_spectral_norm_after_training_forwards(device):
    plan = resolve("linear(32, spectral_norm=True)", input_shape=("B", 24), output_shape=("B", 32))
    model = build(plan, device=device, initialization_seed=11)
    module = model[plan.nodes[0].id]
    for _ in range(5):
        model(x=torch.randn(4, 24, device=device))
    assert largest_singular_value(module.weight) == pytest.approx(1.0, abs=1e-3)
    # The unnormalized parameter is untouched and generally far from unit norm.
    assert largest_singular_value(module.parametrizations.weight.original) != pytest.approx(1.0, abs=1e-3)


def test_power_iteration_updates_only_in_training_mode(device):
    plan = resolve("linear(16, spectral_norm=True)", input_shape=("B", 12), output_shape=("B", 16))
    model = build(plan, device=device, initialization_seed=13)
    module = model[plan.nodes[0].id]
    x = torch.randn(4, 12, device=device)

    before = module.parametrizations.weight[0]._u.clone()
    model(x=x)
    assert not torch.equal(before, module.parametrizations.weight[0]._u)

    model.eval()
    frozen = module.parametrizations.weight[0]._u.clone()
    first = model(x=x)["output"]
    second = model(x=x)["output"]
    assert torch.equal(frozen, module.parametrizations.weight[0]._u)
    torch.testing.assert_close(first, second)


def test_state_dict_round_trips_into_a_rebuilt_network(device):
    source = "linear(16, spectral_norm=True)\nleaky_relu(0.2)\nlinear(spectral_norm=True)"
    plan = resolve(source, input_shape=("B", 12), output_shape=("B", 1))
    trained = build(plan, device=device, initialization_seed=17)
    for _ in range(3):
        trained(x=torch.randn(4, 12, device=device))
    restored = build(plan, device=device, initialization_seed=23)
    assert set(restored.state_dict()) == set(trained.state_dict())
    restored.load_state_dict(trained.state_dict())
    trained.eval()
    restored.eval()
    x = torch.randn(4, 12, device=device)
    torch.testing.assert_close(restored(x=x)["output"], trained(x=x)["output"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device moves are qualified on CUDA")
def test_moving_a_spectrally_normalized_network_to_cuda():
    plan = resolve("linear(16, spectral_norm=True)", input_shape=("B", 12), output_shape=("B", 16))
    model = build(plan, device="cpu", initialization_seed=19)
    model.to("cuda:0")
    module = model[plan.nodes[0].id]
    assert module.parametrizations.weight[0]._u.device.type == "cuda"
    output = model(x=torch.randn(4, 12, device="cuda:0"))["output"]
    assert output.device.type == "cuda" and output.isfinite().all()


def test_construction_settings_must_target_the_reparametrized_name():
    source = 'linear(8, spectral_norm=True, init={"%s": 0.5}, trainable={"%s": False})' % (ORIGINAL, ORIGINAL)
    plan = resolve(source, input_shape=("B", 6), output_shape=("B", 8))
    module = build(plan, device="cpu", initialization_seed=29)[plan.nodes[0].id]
    original = module.parametrizations.weight.original
    assert torch.equal(original, torch.full_like(original, 0.5))
    assert original.requires_grad is False
    assert module.weight.isfinite().all()

    for settings in ('init={"weight": 0.5}', 'trainable={"weight": False}'):
        stale = resolve(f"linear(8, spectral_norm=True, {settings})",
                        input_shape=("B", 6), output_shape=("B", 8))
        with pytest.raises(HNDLError, match="not a registered parameter"):
            build(stale, device="cpu")
