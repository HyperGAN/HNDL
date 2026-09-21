"""Numerics, inference, and error codes for the ``resblock`` operator."""

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.torch import build, parameter_counts

from .conftest import DEVICES

TORCHVISION_BASIC_BLOCK_64 = 73_984


class BasicBlock(nn.Module):
    """A handwritten torchvision-style ``BasicBlock`` used as the reference."""

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 1 and inplanes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


def _randomize(module):
    """Give every normalization non-trivial affine parameters and running stats."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.BatchNorm2d):
                child.weight.uniform_(0.5, 1.5)
                child.bias.uniform_(-0.5, 0.5)
                child.running_mean.uniform_(-1, 1)
                child.running_var.uniform_(0.5, 2.0)
                child.num_batches_tracked.fill_(7)


def _copy_into_reference(block, reference):
    state = {name.replace("norm", "bn").replace("shortcut", "downsample"): value
             for name, value in block.state_dict().items()}
    reference.load_state_dict(state)


def _single(source, input_shape, output_shape, device, seed=11):
    plan = resolve(source, input_shape=input_shape, output_shape=output_shape)
    model = build(plan, device=device, initialization_seed=seed)
    return plan, model, model["n0"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("in_channels,out_channels,stride,extent", [(8, 8, 1, 9), (8, 16, 1, 6), (4, 12, 2, 7),
                                                                    (16, 16, 2, 8)])
def test_matches_a_handwritten_basic_block_in_eval_mode(device, in_channels, out_channels, stride, extent):
    expected_extent = -(-extent // stride)
    _, _, block = _single(f"resblock({out_channels}, stride={stride})", ("B", in_channels, extent, extent),
                          ("B", out_channels, expected_extent, expected_extent), device)
    _randomize(block)
    reference = BasicBlock(in_channels, out_channels, stride).to(device)
    _copy_into_reference(block, reference)
    block.eval()
    reference.eval()

    x = torch.randn(3, in_channels, extent, extent, device=device, requires_grad=True)
    mirror = x.detach().clone().requires_grad_()
    actual, wanted = block(x), reference(mirror)
    assert tuple(actual.shape) == (3, out_channels, expected_extent, expected_extent)
    torch.testing.assert_close(actual, wanted)
    actual.square().mean().backward()
    wanted.square().mean().backward()
    torch.testing.assert_close(x.grad, mirror.grad)
    for name, parameter in block.named_parameters():
        mirrored = dict(reference.named_parameters())[name.replace("norm", "bn").replace("shortcut", "downsample")]
        torch.testing.assert_close(parameter.grad, mirrored.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_train_mode_uses_batch_statistics_and_updates_running_stats(device):
    _, _, block = _single("resblock(8)", ("B", 8, 6, 6), ("B", 8, 6, 6), device)
    _randomize(block)
    x = torch.randn(4, 8, 6, 6, device=device) * 3 + 2

    block.train()
    before = block.norm1.running_mean.clone()
    trained = block(x)
    assert not torch.equal(before, block.norm1.running_mean)
    assert block.norm1.num_batches_tracked.item() == 8

    block.eval()
    stats = block.norm1.running_mean.clone()
    evaluated = block(x)
    assert torch.equal(stats, block.norm1.running_mean)
    assert not torch.allclose(trained, evaluated)


@pytest.mark.parametrize("stride,extent,expected", [(1, 8, 8), (2, 8, 4), (2, 7, 4), (2, 9, 5), (3, 9, 3)])
def test_stride_maps_spatial_extents_like_ceil_division(stride, extent, expected):
    plan = resolve(f"resblock(6, stride={stride})", input_shape=("B", 3, extent, extent),
                   output_shape=("B", 6, expected, expected))
    assert plan.nodes[0].output_shapes["out"] == ("B", 6, expected, expected)
    model = build(plan, device="cpu")
    assert model(x=torch.randn(2, 3, extent, extent))["output"].shape == (2, 6, expected, expected)


def test_channel_change_or_stride_adds_a_projection_shortcut():
    _, _, same = _single("resblock(8)", ("B", 8, 8, 8), ("B", 8, 8, 8), "cpu")
    assert isinstance(same.shortcut, nn.Identity)

    _, _, wider = _single("resblock(16)", ("B", 8, 8, 8), ("B", 16, 8, 8), "cpu")
    assert isinstance(wider.shortcut, nn.Sequential)
    assert wider.shortcut[0].kernel_size == (1, 1) and wider.shortcut[0].stride == (1, 1)
    assert wider.shortcut[0].in_channels == 8 and wider.shortcut[0].out_channels == 16

    _, _, strided = _single("resblock(8, stride=2)", ("B", 8, 8, 8), ("B", 8, 4, 4), "cpu")
    assert isinstance(strided.shortcut, nn.Sequential)
    assert strided.shortcut[0].stride == (2, 2)


@pytest.mark.parametrize("norm,expected", [("batch_norm", nn.BatchNorm2d), ("group_norm", nn.GroupNorm),
                                           ("none", nn.Identity)])
def test_norm_choice_selects_the_normalization_and_convolution_bias(norm, expected):
    _, _, block = _single(f'resblock(8, norm="{norm}", groups=4)', ("B", 4, 6, 6), ("B", 8, 6, 6), "cpu")
    assert isinstance(block.norm1, expected) and isinstance(block.norm2, expected)
    assert isinstance(block.shortcut[1], expected)
    assert (block.conv1.bias is not None) == (norm == "none")
    if norm == "group_norm":
        assert block.norm1.num_groups == 4
    x = torch.randn(2, 4, 6, 6)
    assert block(x).shape == (2, 8, 6, 6)


def test_norm_none_is_a_plain_residual_convolution_block():
    _, _, block = _single('resblock(8, norm="none")', ("B", 8, 6, 6), ("B", 8, 6, 6), "cpu")
    with torch.no_grad():
        block.conv2.weight.zero_()
        block.conv2.bias.zero_()
    x = torch.randn(2, 8, 6, 6)
    torch.testing.assert_close(block(x), torch.relu(x))


def test_parameter_count_matches_the_torchvision_basic_block():
    plan = resolve("resblock(64)", input_shape=("B", 64, 8, 8), output_shape=("B", 64, 8, 8))
    assert parameter_counts(plan) == {"n0": TORCHVISION_BASIC_BLOCK_64}
    assert sum(p.numel() for p in BasicBlock(64, 64).parameters()) == TORCHVISION_BASIC_BLOCK_64
    torchvision = pytest.importorskip("torchvision")
    upstream = torchvision.models.resnet.BasicBlock(64, 64)
    assert sum(p.numel() for p in upstream.parameters()) == TORCHVISION_BASIC_BLOCK_64


def test_shapes_and_channels_are_inferred_in_both_directions():
    forward = resolve("resblock(16, stride=2)\nflatten()\nlinear()", input_shape=("B", 3, 8, 8),
                      output_shape=("B", 10))
    assert forward.nodes[0].args["in_channels"] == 3 and forward.nodes[0].output_shapes["out"] == ("B", 16, 4, 4)

    backward = resolve("resblock()", input_shape=("B", 8, 12, 12), output_shape=("B", 24, 12, 12))
    assert backward.nodes[0].args["out_channels"] == 24

    # Stride 1 inverts exactly, so the seed extent flows back through reshape.
    seeded = resolve("linear()\nreshape(4)\nresblock(4)", input_shape=("B", 16), output_shape=("B", 4, 5, 5))
    assert seeded.nodes[0].args["out_features"] == 100

    # Stride 2 admits two input extents, so the inverse stays ambiguous.
    with pytest.raises(HNDLError, match="E_AMBIGUOUS"):
        resolve("linear()\nreshape(4)\nresblock(4, stride=2)", input_shape=("B", 16), output_shape=("B", 4, 5, 5))


def test_invalid_arguments_and_shapes_report_their_codes():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*group_norm"):
        resolve('resblock(12, norm="group_norm", groups=8)', input_shape=("B", 4, 6, 6),
                output_shape=("B", 12, 6, 6))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve('resblock(8, norm="layer_norm")', input_shape=("B", 4, 6, 6), output_shape=("B", 8, 6, 6))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("resblock(8, stride=0)", input_shape=("B", 4, 6, 6), output_shape=("B", 8, 6, 6))
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve("resblock(0)", input_shape=("B", 4, 6, 6), output_shape=("B", 8, 6, 6))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("resblock(8, stride=2)", input_shape=("B", 4, 8, 8), output_shape=("B", 8, 8, 8))
    with pytest.raises(HNDLError, match="E_CONSTRAINT"):
        resolve("resblock(8)", input_shape=("B", 4, 16), output_shape=("B", 8, 16))
