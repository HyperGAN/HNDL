"""Spectral normalization on the two-dimensional convolution."""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrizations

from hndl import resolve
from hndl.torch import build, parameter_counts

DISCRIMINATOR = ('conv(16, policy="down2", spectral_norm=True)\nleaky_relu(0.2)\n'
                 'conv(32, policy="down2", spectral_norm=True)\nflatten()\nlinear(spectral_norm=True)')


def largest_singular_value(weight):
    return torch.linalg.matrix_norm(weight.detach().reshape(weight.shape[0], -1), 2).item()


def test_forward_and_gradients_match_torch_spectral_norm(device):
    plan = resolve("conv(8, kernel_size=3, padding=1, spectral_norm=True)",
                   input_shape=("B", 4, 12, 12), output_shape=("B", 8, 12, 12))
    model = build(plan, device=device, initialization_seed=7)
    module = model[plan.nodes[0].id]
    assert isinstance(module, nn.Conv2d)
    reference = parametrizations.spectral_norm(nn.Conv2d(4, 8, 3, padding=1)).to(device)
    reference.load_state_dict(module.state_dict())

    x = torch.randn(3, 4, 12, 12, device=device)
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


def test_disabled_spectral_norm_is_a_plain_convolution(device):
    plan = resolve("conv(8, kernel_size=3, padding=1)",
                   input_shape=("B", 4, 12, 12), output_shape=("B", 8, 12, 12))
    assert plan.nodes[0].args["spectral_norm"] is False
    model = build(plan, device=device, initialization_seed=5)
    module = model[plan.nodes[0].id]
    assert sorted(name for name, _ in module.named_parameters()) == ["bias", "weight"]
    reference = nn.Conv2d(4, 8, 3, padding=1).to(device)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(2, 4, 12, 12, device=device)
    torch.testing.assert_close(model(x=x)["output"], reference(x))


def test_effective_kernel_has_unit_spectral_norm_after_training_forwards(device):
    plan = resolve('conv(32, policy="down2", spectral_norm=True)',
                   input_shape=("B", 8, 16, 16), output_shape=("B", 32, 8, 8))
    model = build(plan, device=device, initialization_seed=11)
    module = model[plan.nodes[0].id]
    for _ in range(20):
        model(x=torch.randn(2, 8, 16, 16, device=device))
    # Power iteration underestimates sigma, so the effective norm approaches 1 from above.
    assert largest_singular_value(module.weight) >= 1.0 - 1e-5
    assert largest_singular_value(module.weight) == pytest.approx(1.0, abs=1e-2)
    assert largest_singular_value(module.parametrizations.weight.original) != pytest.approx(1.0, abs=1e-3)


def test_backward_shape_inference_is_unchanged_by_spectral_norm():
    """The declaration adds registered state, never a shape fact."""
    source = 'linear()\nreshape(32)\nconv(16, policy="down2", spectral_norm=True)\nconv(3, policy="down2")'
    plan = resolve(source, input_shape=("B", 16), output_shape=("B", 3, 8, 8))
    assert plan.nodes[1].args["shape"] == (32, 32, 32)
    assert plan.nodes[2].input_shapes["x"] == ("B", 32, 32, 32)
    assert plan.nodes[2].output_shapes["out"] == ("B", 16, 16, 16)
    assert plan.nodes[2].args["in_channels"] == 32

    plain = resolve(source.replace(", spectral_norm=True", ""),
                    input_shape=("B", 16), output_shape=("B", 3, 8, 8))
    assert [dict(n.input_shapes) for n in plan.nodes] == [dict(n.input_shapes) for n in plain.nodes]
    assert [dict(n.output_shapes) for n in plan.nodes] == [dict(n.output_shapes) for n in plain.nodes]


def test_meta_probe_counts_the_reparametrized_state(device):
    plan = resolve(DISCRIMINATOR, input_shape=("B", 3, 16, 16), output_shape=("B", 1))
    model = build(plan, device=device, initialization_seed=3)
    counts = parameter_counts(plan)
    assert counts == {node.id: sum(p.numel() for p in model[node.id].parameters()) for node in plan.nodes}
    assert model.build_receipt["state_bytes"] == sum(
        t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))


def test_discriminator_trains_and_keeps_every_layer_one_lipschitz(device):
    plan = resolve(DISCRIMINATOR, input_shape=("B", 3, 16, 16), output_shape=("B", 1))
    model = build(plan, device=device, initialization_seed=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x=torch.randn(4, 3, 16, 16, device=device))["output"]
        output.square().mean().backward()
        optimizer.step()
    for _ in range(20):  # let the power iteration catch up with the updated weights
        model(x=torch.randn(4, 3, 16, 16, device=device))
    normalized = [module for module in model.nodes.values() if hasattr(module, "parametrizations")]
    assert len(normalized) == 3
    for module in normalized:
        assert largest_singular_value(module.weight) == pytest.approx(1.0, abs=2e-2)
