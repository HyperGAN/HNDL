"""Bilinear/nearest sampling, positional argument rejection, and extra_repr for ``grid_sample``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, Registry, resolve
from hndl.operators.grid_sample import GridSample
from hndl.torch import build


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_parity_and_gradients(device, mode, padding_mode, align_corners):
    module = GridSample(mode=mode, padding_mode=padding_mode, align_corners=align_corners).to(device)
    image = torch.randn(2, 3, 6, 6, device=device, requires_grad=True)
    coords = (torch.randn(2, 4, 5, 2, device=device) * 1.5).requires_grad_()

    actual = module(image, coords)
    expected = F.grid_sample(image, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    assert actual.shape == (2, 3, 4, 5)
    torch.testing.assert_close(actual, expected)

    grad_out = torch.randn_like(actual)
    gi_act, gc_act = torch.autograd.grad(actual, (image, coords), grad_outputs=grad_out, retain_graph=True)
    gi_exp, gc_exp = torch.autograd.grad(expected, (image, coords), grad_outputs=grad_out)
    torch.testing.assert_close(gi_act, gi_exp)
    torch.testing.assert_close(gc_act, gc_exp)


def test_grid_sample_positional_args_rejected():
    reg = Registry.builtins()
    op = reg.get("grid_sample")
    for opt_name in ("mode", "padding_mode", "align_corners"):
        assert op.args[opt_name].positional is False, f"{opt_name} must have positional=False"


def test_grid_sample_extra_repr():
    module = GridSample(mode="nearest", padding_mode="border", align_corners=True)
    rep = module.extra_repr()
    assert "mode='nearest'" in rep
    assert "padding_mode='border'" in rep
    assert "align_corners=True" in rep


def test_grid_sample_built_plan(device):
    source = (
        "g = coordinate_grid(x, 8, 8)\n"
        "grid_sample(x, g, mode='bilinear')"
    )
    plan = resolve(source, input_shape=("B", 3, 4, 4), output_shape=("B", 3, 8, 8))
    model = build(plan, device=device)
    x = torch.randn(2, 3, 4, 4, device=device)
    out = model(x=x)["output"]
    assert out.shape == (2, 3, 8, 8)


def test_grid_sample_invalid_grid_rank():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("grid_sample(a, b)",
                input_shape={"a": ("B", 3, 4, 4), "b": ("B", 8, 8, 3)},
                output_shape=("B", 3, 8, 8))
