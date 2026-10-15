"""Frequencies, determinism, derivative checks, and extra_repr for ``fourier_features``."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.fourier_features import FourierFeatures
from hndl.torch import build


@pytest.mark.parametrize("num_frequencies,d_in", [(8, 2), (16, 3), (32, 4)])
def test_fourier_features_parity_and_derivatives(device, num_frequencies, d_in):
    torch.manual_seed(42)
    module = FourierFeatures(num_frequencies=num_frequencies, scale=8.0, D=d_in).to(device)
    x = torch.randn(2, 5, d_in, device=device, requires_grad=True)

    state = torch.get_rng_state()
    actual = module(x)
    assert torch.equal(torch.get_rng_state(), state), "Forward pass must not advance the RNG"

    phases = x @ module.frequencies.t()
    expected = torch.cat((phases.sin(), phases.cos()), dim=-1)
    assert actual.shape == (2, 5, 2 * num_frequencies)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gx_act, = torch.autograd.grad(actual, (x,), grad_outputs=grad_out, retain_graph=True)
    gx_exp, = torch.autograd.grad(expected, (x,), grad_outputs=grad_out)
    torch.testing.assert_close(gx_act, gx_exp)


def test_fourier_features_extra_repr():
    module = FourierFeatures(num_frequencies=16, scale=8.0, D=2)
    rep = module.extra_repr()
    assert "num_frequencies=16" in rep
    assert "scale=8.0" in rep


def test_fourier_features_built_plan(device):
    plan = resolve("fourier_features(8, scale=8.0)", input_shape=("B", 16, 2), output_shape=("B", 16, 16))
    model = build(plan, device=device)
    x = torch.randn(2, 16, 2, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 16, 16)


def test_fourier_features_invalid_arguments():
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("fourier_features(0)", input_shape=("B", 3, 2), output_shape=("B", 3, 4))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("fourier_features(2, scale=0.0)", input_shape=("B", 3, 2), output_shape=("B", 3, 4))
