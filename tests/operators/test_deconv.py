"""Spectral normalization on the transposed convolution.

PyTorch normalizes a transposed convolution over its *output*-channel axis
(``dim=1``), which is the case these tests pin.
"""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrizations

from hndl import resolve
from hndl.torch import build


def transposed_singular_value(weight):
    """Largest singular value of the [C_out, C_in * kH * kW] matrix view."""
    matrix = weight.detach().transpose(0, 1).reshape(weight.shape[1], -1)
    return torch.linalg.matrix_norm(matrix, 2).item()


def test_forward_and_gradients_match_torch_spectral_norm(device):
    plan = resolve('deconv(8, policy="up2", spectral_norm=True)',
                   input_shape=("B", 4, 8, 8), output_shape=("B", 8, 16, 16))
    model = build(plan, device=device, initialization_seed=7)
    module = model[plan.nodes[0].id]
    assert isinstance(module, nn.ConvTranspose2d)
    reference = parametrizations.spectral_norm(
        nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1)).to(device)
    reference.load_state_dict(module.state_dict())
    assert reference.parametrizations.weight[0].dim == 1

    x = torch.randn(3, 4, 8, 8, device=device)
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


def test_disabled_spectral_norm_is_a_plain_transposed_convolution(device):
    plan = resolve('deconv(8, policy="up2")', input_shape=("B", 4, 8, 8), output_shape=("B", 8, 16, 16))
    assert plan.nodes[0].args["spectral_norm"] is False
    model = build(plan, device=device, initialization_seed=5)
    module = model[plan.nodes[0].id]
    assert sorted(name for name, _ in module.named_parameters()) == ["bias", "weight"]
    reference = nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1).to(device)
    reference.load_state_dict(module.state_dict())
    x = torch.randn(2, 4, 8, 8, device=device)
    torch.testing.assert_close(model(x=x)["output"], reference(x))


def test_effective_kernel_has_unit_spectral_norm_after_training_forwards(device):
    plan = resolve('deconv(16, policy="up2", spectral_norm=True)',
                   input_shape=("B", 8, 8, 8), output_shape=("B", 16, 16, 16))
    model = build(plan, device=device, initialization_seed=11)
    module = model[plan.nodes[0].id]
    for _ in range(20):
        model(x=torch.randn(2, 8, 8, 8, device=device))
    assert transposed_singular_value(module.weight) >= 1.0 - 1e-5
    assert transposed_singular_value(module.weight) == pytest.approx(1.0, abs=1e-2)


def test_backward_shape_inference_is_unchanged_by_spectral_norm():
    source = 'linear()\nreshape(32)\ndeconv(16, policy="up2", spectral_norm=True)\ndeconv(3, policy="up2")'
    plan = resolve(source, input_shape=("B", 16), output_shape=("B", 3, 16, 16))
    assert plan.nodes[1].args["shape"] == (32, 4, 4)
    assert plan.nodes[2].output_shapes["out"] == ("B", 16, 8, 8)
    assert plan.nodes[2].args["in_channels"] == 32

    plain = resolve(source.replace(", spectral_norm=True", ""),
                    input_shape=("B", 16), output_shape=("B", 3, 16, 16))
    assert [dict(n.output_shapes) for n in plan.nodes] == [dict(n.output_shapes) for n in plain.nodes]


def test_state_dict_round_trips_and_evaluation_is_frozen(device):
    plan = resolve('deconv(8, policy="up2", spectral_norm=True)',
                   input_shape=("B", 4, 8, 8), output_shape=("B", 8, 16, 16))
    trained = build(plan, device=device, initialization_seed=17)
    for _ in range(3):
        trained(x=torch.randn(2, 4, 8, 8, device=device))
    restored = build(plan, device=device, initialization_seed=23)
    restored.load_state_dict(trained.state_dict())
    trained.eval()
    restored.eval()
    x = torch.randn(2, 4, 8, 8, device=device)
    torch.testing.assert_close(restored(x=x)["output"], trained(x=x)["output"])
    torch.testing.assert_close(restored(x=x)["output"], restored(x=x)["output"])
