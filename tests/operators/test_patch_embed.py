"""Numerics, token ordering, bidirectional inference, and error codes for ``patch_embed``."""

import pytest
import torch
import torch.nn.functional as F

from hndl import HNDLError, resolve
from hndl.torch import DTYPES, build


def _module(source, input_shape, output_shape, device, dtype="float32"):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape, dtype=dtype)
    model = build(plan, device=device, initialization_seed=11)
    return plan, model


@pytest.mark.parametrize("channels,height,width,patch,dim", [(3, 16, 16, 4, 32), (1, 8, 12, 4, 6), (5, 6, 6, 3, 8)])
def test_matches_an_unfold_reference_forward_and_backward(channels, height, width, patch, dim, device):
    source = f"patch_embed({dim}, {patch})"
    tokens = (height // patch) * (width // patch)
    _, model = _module(source, ("B", channels, height, width), ("B", tokens, dim), device)
    layer = model["n0"]

    x = torch.randn(3, channels, height, width, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual = model(x=x)["output"]

    # An independent implementation: gather each patch with unfold, then apply
    # the projection as a matmul over the flattened [C, patch, patch] window.
    patches = F.unfold(mirror, kernel_size=patch, stride=patch).transpose(1, 2)
    expected = patches @ layer.proj.weight.reshape(dim, -1).t() + layer.proj.bias

    assert tuple(actual.shape) == (3, tokens, dim)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad, rtol=1e-5, atol=1e-5)


def test_tokens_are_row_major_patches_of_the_image(device):
    _, model = _module("patch_embed(4, 2)", ("B", 2, 4, 6), ("B", 6, 4), device)
    layer = model["n0"]
    x = torch.randn(2, 2, 4, 6, device=device)
    out = model(x=x)["output"]
    weight = layer.proj.weight.reshape(4, -1)
    for row in range(2):
        for column in range(3):
            window = x[:, :, row * 2:row * 2 + 2, column * 2:column * 2 + 2].reshape(2, -1)
            torch.testing.assert_close(out[:, row * 3 + column], window @ weight.t() + layer.proj.bias)


@pytest.mark.parametrize("source,input_shape,output_shape", [
    ("patch_embed(8, 3)", ("B", 3, 16, 16), ("B", 25, 8)),
    ("patch_embed(8, 4)", ("B", 3, 8, 10), ("B", 4, 8)),
])
def test_a_patch_size_that_does_not_tile_the_image_is_a_constraint_error(source, input_shape, output_shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*must divide the input"):
        resolve(source, input_shape=input_shape, output_shape=output_shape)


def test_an_impossible_token_count_is_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*9 tokens cannot be split"):
        resolve("linear()\nreshape(3, 8)\npatch_embed(8, 4)", input_shape=("B", 16), output_shape=("B", 9, 8))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("patch_embed(8, 4)", input_shape=("B", 3, 8, 8), output_shape=("B", 5, 8))


def test_dim_and_in_channels_are_inferred_from_the_neighbouring_contracts():
    plan = resolve("patch_embed(patch_size=4)", input_shape=("B", 5, 8, 16), output_shape=("B", 8, 24))
    node = plan.nodes[0]
    assert node.args == {"dim": 24, "patch_size": 4, "in_channels": 5}
    assert node.provenance["dim"] == "inferred" and node.provenance["in_channels"] == "inferred"
    assert node.output_shapes["out"] == ("B", 8, 24)


def test_a_known_token_count_completes_one_missing_spatial_extent():
    plan = resolve("linear()\nreshape(3, 8)\npatch_embed(8, 4)", input_shape=("B", 16), output_shape=("B", 8, 8))
    assert plan.nodes[1].output_shapes["out"] == ("B", 3, 8, 16)
    assert plan.nodes[0].args["out_features"] == 384


def test_a_token_count_alone_does_not_guess_the_image_shape():
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("reshape(3)\npatch_embed(8, 4)", input_shape=("B", 384), output_shape=("B", 8, 8))


def test_a_non_image_input_is_rejected():
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("patch_embed(8, 4)", input_shape=("B", 3, 16), output_shape=("B", 4, 8))


@pytest.mark.parametrize("patch_size", [0, -2])
def test_a_non_positive_patch_size_is_an_argument_error(patch_size):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(f"patch_embed(8, {patch_size})", input_shape=("B", 3, 8, 8), output_shape=("B", 4, 8))


def test_patch_size_is_required():
    with pytest.raises(HNDLError):
        resolve("patch_embed(8)", input_shape=("B", 3, 8, 8), output_shape=("B", 4, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
def test_bfloat16_on_cuda_matches_the_float32_result_closely():
    source = "patch_embed(16, 4)\nlinear()"
    shapes = (("B", 3, 8, 8), ("B", 4, 6))
    _, reference = _module(source, *shapes, "cuda:0")
    _, model = _module(source, *shapes, "cuda:0", dtype="bfloat16")
    assert all(p.dtype == DTYPES["bfloat16"] for p in model.parameters())
    x = torch.randn(2, 3, 8, 8, device="cuda:0")
    out = model(x=x.to(DTYPES["bfloat16"]))["output"]
    assert out.dtype == DTYPES["bfloat16"] and tuple(out.shape) == (2, 4, 6)
    torch.testing.assert_close(out.float(), reference(x=x)["output"], rtol=3e-2, atol=3e-2)
