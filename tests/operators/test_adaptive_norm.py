"""Numerics, shape propagation, and error checks for the ``adaptive_norm`` operator."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.operators.adaptive_norm import AdaptiveNorm
from hndl.torch import build


def reference_adaptive_norm(x, params, eps=1e-5):
    """Ground truth reference implementation for adaptive instance normalization."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = (x - mean).square().mean(dim=(2, 3), keepdim=True)
    normalized = (x - mean) * torch.rsqrt(var + eps)
    channels = x.shape[1]
    delta_gamma, beta = params[:, :channels], params[:, channels:]
    return (1 + delta_gamma[:, :, None, None]) * normalized + beta[:, :, None, None]


@pytest.mark.parametrize("channels,height,width", [(4, 8, 8), (8, 4, 6), (16, 2, 2)])
def test_adaptive_norm_parity_and_gradients(device, channels, height, width):
    batch = 2
    module = AdaptiveNorm(eps=1e-5).to(device)
    x = torch.randn(batch, channels, height, width, device=device, requires_grad=True)
    params = torch.randn(batch, 2 * channels, device=device, requires_grad=True)

    actual = module(x, params)
    expected = reference_adaptive_norm(x, params, eps=1e-5)
    assert actual.shape == (batch, channels, height, width)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gx_act, gp_act = torch.autograd.grad(actual, (x, params), grad_outputs=grad_out, retain_graph=True)
    gx_exp, gp_exp = torch.autograd.grad(expected, (x, params), grad_outputs=grad_out)
    torch.testing.assert_close(gx_act, gx_exp)
    torch.testing.assert_close(gp_act, gp_exp)


def test_adaptive_norm_zero_params_matches_instance_norm(device):
    batch, channels, h, w = 2, 4, 8, 8
    module = AdaptiveNorm(eps=1e-5).to(device)
    x = torch.randn(batch, channels, h, w, device=device)
    params = torch.zeros(batch, 2 * channels, device=device)

    actual = module(x, params)
    expected = F.instance_norm(x, eps=1e-5)
    torch.testing.assert_close(actual, expected)


def test_adaptive_norm_built_plan(device):
    source = (
        "z1, z2 = split(32)\n"
        "linear(z1, 16)\n"
        "feat = reshape(8, 2, 1)\n"
        "adaptive_norm(feat, z2)"
    )
    plan = resolve(source, input_shape=("B", 48), output_shape=("B", 8, 2, 1))
    model = build(plan, device=device)
    x = torch.randn(2, 48, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 8, 2, 1)


def test_adaptive_norm_extra_repr():
    norm = AdaptiveNorm(eps=1e-5)
    assert "eps=1e-05" in norm.extra_repr()


def test_adaptive_norm_mismatched_params_shape():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("adaptive_norm(feat, style)",
                input_shape={"feat": ("B", 8, 4, 4), "style": ("B", 10)},
                output_shape=("B", 8, 4, 4))
