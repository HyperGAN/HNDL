"""Numerics, bidirectional inference, and error codes for ``upsample``."""

import warnings

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import build

MODES = ["nearest", "bilinear", "bicubic"]


def upsampled(source, input_shape, output_shape, device, batch=3):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=7)
    x = torch.randn(batch, *input_shape[1:], device=device)
    return plan, model, x


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scale,extent", [(2, 5), (3, 4), (4, 2)])
def test_matches_torch_interpolate_in_every_mode(mode, scale, extent, device):
    source = f'upsample({scale}, mode="{mode}")'
    shape = ("B", 5, extent, extent + 1)
    target = ("B", 5, extent * scale, (extent + 1) * scale)
    _, model, x = upsampled(source, shape, target, device)
    align_corners = None if mode == "nearest" else False
    expected = F.interpolate(x, scale_factor=float(scale), mode=mode, align_corners=align_corners)
    torch.testing.assert_close(model(x=x)["output"], expected)


def test_nearest_repeats_each_pixel_into_a_block(device):
    _, model, x = upsampled("upsample(3)", ("B", 2, 4, 4), ("B", 2, 12, 12), device)
    expected = x.repeat_interleave(3, dim=2).repeat_interleave(3, dim=3)
    torch.testing.assert_close(model(x=x)["output"], expected)


@pytest.mark.parametrize("align_corners", [False, True])
def test_align_corners_selects_the_sampling_grid(align_corners, device):
    source = f"upsample(2, mode=\"bilinear\", align_corners={align_corners})"
    _, model, x = upsampled(source, ("B", 3, 4, 4), ("B", 3, 8, 8), device)
    expected = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=align_corners)
    torch.testing.assert_close(model(x=x)["output"], expected)
    other = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=not align_corners)
    assert not torch.allclose(expected, other)


def test_gradients_flow_back_through_the_interpolation(device):
    _, model, x = upsampled('upsample(2, mode="bilinear")', ("B", 2, 4, 4), ("B", 2, 8, 8), device)
    x = x.requires_grad_()
    mirror = x.detach().clone().requires_grad_()
    model(x=x)["output"].square().mean().backward()
    F.interpolate(mirror, scale_factor=2.0, mode="bilinear", align_corners=False).square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_scale_factor_one_is_the_identity(device):
    _, model, x = upsampled("upsample(1)", ("B", 3, 5, 5), ("B", 3, 5, 5), device)
    torch.testing.assert_close(model(x=x)["output"], x)


def test_forward_inference_fixes_the_output_extents():
    plan = resolve("upsample(3)\nconv(4, kernel_size=3, padding=1)",
                   input_shape=("B", 2, 5, 7), output_shape=("B", 4, 15, 21))
    assert plan.nodes[0].output_shapes["out"] == ("B", 2, 15, 21)


def test_backward_inference_solves_the_producing_layer():
    plan = resolve("linear()\nreshape(16)\nupsample(2)", input_shape=("B", 32), output_shape=("B", 16, 8, 8))
    assert plan.nodes[0].args["out_features"] == 16 * 4 * 4
    assert plan.nodes[1].args["shape"] == (16, 4, 4)
    assert plan.nodes[2].input_shapes["x"] == ("B", 16, 4, 4)


def test_channels_pass_through_unchanged_in_both_directions():
    plan = resolve("upsample(2)", input_shape=("B", 7, 3, 3), output_shape=("B", 7, 6, 6))
    assert plan.nodes[0].output_shapes["out"] == ("B", 7, 6, 6)
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("upsample(2)", input_shape=("B", 7, 3, 3), output_shape=("B", 5, 6, 6))


@pytest.mark.parametrize("target", [("B", 16, 9, 8), ("B", 16, 8, 9)])
def test_non_divisible_target_extent_is_a_constraint_error(target):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*multiple of scale_factor"):
        resolve("linear()\nreshape(16)\nupsample(2)", input_shape=("B", 32), output_shape=target)


def test_contradictory_output_shape_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("upsample(2)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 16, 16))


def test_rank_two_input_is_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("upsample(2)", input_shape=("B", 16), output_shape=("B", 16))


def test_align_corners_with_nearest_is_an_argument_error():
    with pytest.raises(HNDLError, match="E_ARGUMENT.*align_corners"):
        resolve("upsample(2, align_corners=True)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 8, 8))


@pytest.mark.parametrize("source", ['upsample(2, mode="trilinear")', "upsample(0)", "upsample(-1)"])
def test_invalid_arguments_are_argument_errors(source):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(source, input_shape=("B", 3, 4, 4), output_shape=("B", 3, 8, 8))


def test_nearest_runs_without_an_align_corners_warning(device):
    plan = resolve("upsample(2)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 8, 8))
    model = build(plan, device=device)
    assert model["n0"].align_corners is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(x=torch.randn(2, 3, 4, 4, device=device))
