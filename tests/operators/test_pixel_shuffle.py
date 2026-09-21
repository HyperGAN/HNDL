"""Numerics, bidirectional inference, and error codes for ``pixel_shuffle``."""

import pytest
import torch
from torch.nn import functional as F

from hndl import HNDLError, resolve
from hndl.torch import build


def shuffled(source, input_shape, output_shape, device, batch=3):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=11)
    x = torch.randn(batch, *input_shape[1:], device=device)
    return plan, model, x


@pytest.mark.parametrize("factor,channels,height,width", [(2, 3, 4, 5), (3, 2, 5, 3), (4, 1, 2, 3)])
def test_matches_torch_pixel_shuffle(factor, channels, height, width, device):
    shape = ("B", channels * factor * factor, height, width)
    target = ("B", channels, height * factor, width * factor)
    _, model, x = shuffled(f"pixel_shuffle({factor})", shape, target, device)
    torch.testing.assert_close(model(x=x)["output"], F.pixel_shuffle(x, factor))


def test_sub_pixel_layout_is_the_documented_permutation(device):
    factor, channels = 2, 3
    shape = ("B", channels * factor * factor, 4, 4)
    target = ("B", channels, 8, 8)
    _, model, x = shuffled(f"pixel_shuffle({factor})", shape, target, device, batch=2)
    out = model(x=x)["output"]
    for c in range(channels):
        for i in range(factor):
            for j in range(factor):
                torch.testing.assert_close(out[:, c, i::factor, j::factor],
                                           x[:, c * factor * factor + i * factor + j])


def test_round_trips_with_pixel_unshuffle(device):
    _, model, x = shuffled("pixel_shuffle(2)", ("B", 8, 4, 4), ("B", 2, 8, 8), device)
    torch.testing.assert_close(F.pixel_unshuffle(model(x=x)["output"], 2), x)


def test_gradients_match_the_torch_reference(device):
    _, model, x = shuffled("pixel_shuffle(2)", ("B", 12, 3, 3), ("B", 3, 6, 6), device)
    x = x.requires_grad_()
    mirror = x.detach().clone().requires_grad_()
    (model(x=x)["output"] * 3.0).square().mean().backward()
    (F.pixel_shuffle(mirror, 2) * 3.0).square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)


def test_factor_one_is_the_identity(device):
    _, model, x = shuffled("pixel_shuffle(1)", ("B", 5, 4, 4), ("B", 5, 4, 4), device)
    torch.testing.assert_close(model(x=x)["output"], x)


def test_forward_inference_divides_channels_and_scales_extents():
    plan = resolve("pixel_shuffle(3)\nconv(2, kernel_size=3, padding=1)",
                   input_shape=("B", 18, 4, 5), output_shape=("B", 2, 12, 15))
    assert plan.nodes[0].output_shapes["out"] == ("B", 2, 12, 15)


def test_backward_inference_solves_the_producing_convolution():
    plan = resolve("conv(kernel_size=3, padding=1)\npixel_shuffle(2)",
                   input_shape=("B", 3, 8, 8), output_shape=("B", 5, 16, 16))
    assert plan.nodes[0].args["out_channels"] == 20
    assert plan.nodes[1].input_shapes["x"] == ("B", 20, 8, 8)


def test_backward_inference_solves_the_seed_shape():
    plan = resolve("linear()\nreshape(48)\npixel_shuffle(4)", input_shape=("B", 16), output_shape=("B", 3, 8, 8))
    assert plan.nodes[1].args["shape"] == (48, 2, 2)
    assert plan.nodes[0].args["out_features"] == 48 * 2 * 2


def test_non_divisible_channel_count_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divisible by upscale_factor"):
        resolve("pixel_shuffle(2)", input_shape=("B", 6, 4, 4), output_shape=("B", 1, 8, 8))
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*divisible by upscale_factor"):
        resolve("pixel_shuffle(3)\nrelu()", input_shape=("B", 8, 4, 4), output_shape=("B", 1, 12, 12))


@pytest.mark.parametrize("target", [("B", 3, 9, 8), ("B", 3, 8, 9)])
def test_non_divisible_target_extent_is_a_constraint_error(target):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*multiple of upscale_factor"):
        resolve("linear()\nreshape(12)\npixel_shuffle(2)", input_shape=("B", 32), output_shape=target)


def test_contradictory_output_channels_are_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("pixel_shuffle(2)", input_shape=("B", 8, 4, 4), output_shape=("B", 3, 8, 8))


def test_rank_two_input_is_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("pixel_shuffle(2)", input_shape=("B", 16), output_shape=("B", 4))


@pytest.mark.parametrize("source", ["pixel_shuffle(0)", "pixel_shuffle(-2)"])
def test_invalid_factors_are_argument_errors(source):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(source, input_shape=("B", 16, 4, 4), output_shape=("B", 4, 8, 8))


def test_upscale_factor_is_required():
    with pytest.raises(HNDLError):
        resolve("pixel_shuffle()", input_shape=("B", 16, 4, 4), output_shape=("B", 4, 8, 8))
