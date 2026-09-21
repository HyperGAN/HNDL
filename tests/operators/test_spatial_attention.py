"""Numerics, error codes, and the SAGAN equivalence of the ``spatial_attention`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.spatial_attention import SpatialAttention
from hndl.torch import build, parameter_counts

from .conftest import DEVICES

# The hand-composed SAGAN block this operator replaces, kept verbatim so the
# equivalence below stays meaningful after examples/networks/sagan_attention.hndl
# was refactored to a single spatial_attention() call.
HAND_COMPOSED = """
saved = x
f = conv(x, 8, kernel_size=1, name="f")
g = conv(x, 8, kernel_size=1, name="g")
h = conv(x, 64, kernel_size=1, name="h")
fn = reshape(f, 8)
gn = reshape(g, 8)
hn = reshape(h, 64)
ft = transpose(fn, 1, 2)
energy = matmul(ft, gn)
beta = softmax(energy, -1)
bt = transpose(beta, 1, 2)
o = matmul(hn, bt)
o = reshape(o, 64, 16, 16)
o = learned_scale(o, name="gamma")
add(o, saved)
"""

SHAPE = ("B", 64, 16, 16)


def make_module(device, dtype, *, reduction=8, channels=64, seed=0):
    torch.manual_seed(seed)
    module = SpatialAttention(reduction=reduction, C=channels)
    return module.to(device=device, dtype=dtype).eval()


def explicit_spatial_attention(module, x):
    """The block written with broadcast sums, so no batched-matmul kernel is involved."""
    batch, channels, height, width = x.shape
    positions = height * width
    f = module.f_proj(x).reshape(batch, module.query_channels, positions)
    g = module.g_proj(x).reshape(batch, module.query_channels, positions)
    h = module.h_proj(x).reshape(batch, channels, positions)
    energy = (f.unsqueeze(-1) * g.unsqueeze(-2)).sum(1)          # [B, N_query, N_key]
    beta = energy.softmax(-1)
    attended = (h.unsqueeze(-2) * beta.unsqueeze(1)).sum(-1)     # [B, C, N_query]
    return module.gamma * attended.reshape(batch, channels, height, width) + x


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction,channels,extent", [(8, 64, 4), (1, 4, 3), (2, 16, 5), (4, 8, 2)])
def test_matches_the_explicit_block_forward_and_backward(device, reduction, channels, extent):
    dtype = torch.float64 if device == "cpu" else torch.float32
    tolerance = {} if device == "cpu" else {"atol": 1e-5, "rtol": 1e-4}
    module = make_module(device, dtype, reduction=reduction, channels=channels, seed=channels + extent)
    with torch.no_grad():
        # gamma starts at zero, which would make both sides the identity.
        module.gamma.fill_(0.7)
    x = torch.randn(3, channels, extent, extent, device=device, dtype=dtype)
    actual_input = x.detach().clone().requires_grad_()
    expected_input = x.detach().clone().requires_grad_()

    actual = module(actual_input)
    expected = explicit_spatial_attention(module, expected_input)
    torch.testing.assert_close(actual, expected, **tolerance)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, **tolerance)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


@pytest.mark.parametrize("device", DEVICES)
def test_matches_the_hand_composed_sagan_block(device):
    """The refactor is behavior-preserving: the operator equals the block it replaces."""
    plan = resolve(HAND_COMPOSED, input_shape=SHAPE, output_shape=SHAPE)
    composed = build(plan, device=device, initialization_seed=4).eval()
    operator = SpatialAttention(reduction=8, C=64).to(device).eval()
    with torch.no_grad():
        for source, target in (("f", operator.f_proj), ("g", operator.g_proj), ("h", operator.h_proj)):
            target.weight.copy_(composed[source].weight)
            target.bias.copy_(composed[source].bias)
        # A zero gate would hide every difference, so open it on both sides.
        composed["gamma"].gamma.fill_(0.7)
        operator.gamma.copy_(composed["gamma"].gamma)

    x = torch.randn(2, 64, 16, 16, device=device)
    torch.testing.assert_close(operator(x), composed(x=x)["output"])


def test_the_gate_starts_closed_so_the_block_starts_as_the_identity():
    module = make_module("cpu", torch.float64, seed=1)
    x = torch.randn(2, 64, 4, 4, dtype=torch.float64)
    assert float(module.gamma.detach()) == 0.0
    torch.testing.assert_close(module(x), x)


def test_a_single_position_attends_only_to_itself():
    """A 1x1 map softmaxes over one key, so the attention is the value projection."""
    module = make_module("cpu", torch.float64, reduction=4, channels=8, seed=4)
    with torch.no_grad():
        module.gamma.fill_(0.7)
    x = torch.randn(3, 8, 1, 1, dtype=torch.float64)
    torch.testing.assert_close(module(x), 0.7 * module.h_proj(x) + x)


@pytest.mark.parametrize("device", DEVICES)
def test_every_position_can_influence_every_other(device):
    """Unlike a convolution, the block is not local: one corner moves the opposite one."""
    module = make_module(device, torch.float64, reduction=2, channels=8, seed=2)
    with torch.no_grad():
        module.gamma.fill_(1.0)
    x = torch.randn(1, 8, 5, 5, device=device, dtype=torch.float64)
    changed = x.clone()
    changed[:, :, 0, 0] += 3.0
    difference = (module(x) - module(changed))[:, :, 4, 4].abs().max()
    assert difference > 1e-6


def test_parameters_are_three_projections_and_one_gate():
    plan = resolve("spatial_attention()", input_shape=SHAPE, output_shape=SHAPE)
    # Two Conv2d(64, 8, 1) with bias, one Conv2d(64, 64, 1) with bias, plus gamma.
    assert parameter_counts(plan) == {"n0": 2 * (64 * 8 + 8) + (64 * 64 + 64) + 1}
    assert sum(parameter_counts(plan).values()) == 5201
    model = build(plan, device="cpu")
    assert model["n0"].query_channels == 8
    assert sorted(name for name, _ in model["n0"].named_parameters()) == [
        "f_proj.bias", "f_proj.weight", "g_proj.bias", "g_proj.weight",
        "gamma", "h_proj.bias", "h_proj.weight"]


def test_channels_flow_forward_and_backward_through_spatial_attention():
    forward = resolve("spatial_attention()\nglobal_avg_pool()\nlinear()",
                      input_shape=("B", 32, 4, 4), output_shape=("B", 5))
    assert forward.nodes[0].output_shapes["out"] == ("B", 32, 4, 4)

    backward = resolve("conv(kernel_size=1)\nspatial_attention(4)",
                       input_shape=("B", 3, 4, 4), output_shape=("B", 16, 4, 4))
    assert backward.nodes[0].args["out_channels"] == 16
    assert backward.nodes[1].input_shapes["x"] == ("B", 16, 4, 4)


@pytest.mark.parametrize("source,shape,output_shape", [
    ("spatial_attention(5)", ("B", 64, 4, 4), ("B", 64, 4, 4)),
    ("spatial_attention(3)", ("B", 16, 8, 8), ("B", 16, 8, 8)),
    ("conv(kernel_size=1)\nspatial_attention(7)", ("B", 3, 4, 4), ("B", 12, 4, 4)),
])
def test_invalid_reductions_report_a_constraint_error(source, shape, output_shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*must divide the channel width"):
        resolve(source, input_shape=shape, output_shape=output_shape)


def test_direct_construction_guards_the_reduction():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*must divide the channel width"):
        SpatialAttention(reduction=5, C=64)


def test_rank_three_input_is_rejected():
    with pytest.raises(HNDLError):
        resolve("spatial_attention()", input_shape=("B", 64, 16), output_shape=("B", 64, 16))
