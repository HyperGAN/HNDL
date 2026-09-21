"""Numerics, position handling, and error codes for the ``attention`` operator."""

import math

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.operators.attention import Attention
from hndl.torch import build, parameter_counts

from .conftest import DEVICES


def precision(device):
    """Double precision on the CPU; float32 with loose tolerances on fused CUDA kernels."""
    if device == "cpu":
        return torch.float64, {}
    return torch.float32, {"atol": 1e-5, "rtol": 1e-4}


def make_module(device, dtype, *, heads=4, width=32, causal=False, dropout=0.0, bias=True, rope=False, seed=0):
    torch.manual_seed(seed)
    module = Attention(heads=heads, causal=causal, dropout=dropout, bias=bias, rope=rope, D=width)
    return module.to(device=device, dtype=dtype).eval()


def explicit_attention(module, x):
    """softmax(QK^T / sqrt(head_dim) + mask) V written out, with no fused kernel."""
    batch, positions, width = x.shape
    heads, head_dim = module.heads, module.head_dim

    def heads_of(projection):
        return projection(x).view(batch, positions, heads, head_dim).transpose(1, 2)

    q, k, v = heads_of(module.q_proj), heads_of(module.k_proj), heads_of(module.v_proj)
    if module.rope:
        angle = torch.arange(positions, device=x.device, dtype=torch.float64)[:, None] * torch.pow(
            torch.tensor(10000.0, device=x.device, dtype=torch.float64),
            -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float64) / head_dim)[None, :]
        cos = torch.cat((angle, angle), -1).cos().to(x.dtype)[None, None]
        sin = torch.cat((angle, angle), -1).sin().to(x.dtype)[None, None]
        half = head_dim // 2
        q, k = ((t * cos + torch.cat((-t[..., half:], t[..., :half]), -1) * sin) for t in (q, k))
    # Broadcast sums rather than matmuls: the contractions stay readable and no
    # batched-matmul kernel is involved, so this really is an independent check.
    scores = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1) / math.sqrt(head_dim)
    if module.causal:
        scores = scores.masked_fill(torch.ones(positions, positions, dtype=torch.bool, device=x.device).triu(1),
                                    float("-inf"))
    attended = (scores.softmax(-1).unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)
    return module.o_proj(attended.transpose(1, 2).reshape(batch, positions, width))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("heads,width,positions", [(1, 8, 5), (4, 32, 7), (8, 16, 3), (2, 16, 1)])
def test_matches_explicit_softmax_attention_forward_and_backward(device, causal, rope, heads, width, positions):
    dtype, tolerance = precision(device)
    module = make_module(device, dtype, heads=heads, width=width, causal=causal, rope=rope, seed=heads + positions)
    x = torch.randn(3, positions, width, device=device, dtype=dtype)
    actual_input = x.detach().clone().requires_grad_()
    expected_input = x.detach().clone().requires_grad_()

    actual = module(actual_input)
    expected = explicit_attention(module, expected_input)
    torch.testing.assert_close(actual, expected, **tolerance)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, **tolerance)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [False, True])
def test_matches_torch_multihead_attention_with_copied_weights(device, bias):
    heads, width, positions = 4, 24, 9
    module = make_module(device, torch.float32, heads=heads, width=width, bias=bias, seed=11)
    torch_mha = nn.MultiheadAttention(width, heads, bias=bias, batch_first=True)
    torch_mha = torch_mha.to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(torch.cat([module.q_proj.weight, module.k_proj.weight,
                                                  module.v_proj.weight], dim=0))
        torch_mha.out_proj.weight.copy_(module.o_proj.weight)
        if bias:
            torch_mha.in_proj_bias.copy_(torch.cat([module.q_proj.bias, module.k_proj.bias, module.v_proj.bias]))
            torch_mha.out_proj.bias.copy_(module.o_proj.bias)

    x = torch.randn(2, positions, width, device=device)
    expected, _ = torch_mha(x, x, x, need_weights=False)
    torch.testing.assert_close(module(x), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_rope_ties_the_output_to_absolute_positions(device):
    """Plain attention is permutation-equivariant; rotary embeddings break that."""
    dtype, tolerance = precision(device)
    plain = make_module(device, dtype, heads=2, width=16, rope=False, seed=3)
    rotary = make_module(device, dtype, heads=2, width=16, rope=True, seed=3)
    rotary.load_state_dict(plain.state_dict())
    x = torch.randn(2, 6, 16, device=device, dtype=dtype)
    shifted = x.roll(1, dims=1)

    torch.testing.assert_close(plain(shifted), plain(x).roll(1, dims=1), **tolerance)
    difference = (rotary(shifted) - rotary(x).roll(1, dims=1)).abs().max()
    assert difference > 1e-3, "rope must make the output depend on where a token sits"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal", [False, True])
def test_rope_is_a_no_op_for_a_single_position(device, causal):
    plain = make_module(device, torch.float32, heads=2, width=8, rope=False, causal=causal, seed=5)
    rotary = make_module(device, torch.float32, heads=2, width=8, rope=True, causal=causal, seed=5)
    rotary.load_state_dict(plain.state_dict())
    x = torch.randn(4, 1, 8, device=device)
    torch.testing.assert_close(rotary(x), plain(x))


@pytest.mark.parametrize("device", DEVICES)
def test_causal_output_ignores_later_positions(device):
    dtype, tolerance = precision(device)
    module = make_module(device, dtype, heads=2, width=8, causal=True, seed=7)
    x = torch.randn(1, 6, 8, device=device, dtype=dtype)
    later = x.clone()
    later[:, 3:] = torch.randn_like(later[:, 3:])
    torch.testing.assert_close(module(x)[:, :3], module(later)[:, :3], **tolerance)
    assert (module(x)[:, 3:] - module(later)[:, 3:]).abs().max() > 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_applies_in_training_mode_only(device):
    module = make_module(device, torch.float32, heads=2, width=16, dropout=0.5, seed=9)
    x = torch.randn(4, 6, 16, device=device)
    module.eval()
    torch.testing.assert_close(module(x), module(x))
    module.train()
    torch.manual_seed(0)
    first = module(x)
    torch.manual_seed(1)
    assert not torch.allclose(first, module(x))


def test_widths_flow_forward_and_backward_through_attention():
    forward = resolve("attention(4)\nlinear()", input_shape=("B", 8, 32), output_shape=("B", 8, 5))
    assert forward.nodes[0].output_shapes["out"] == ("B", 8, 32)
    assert forward.nodes[1].args["out_features"] == 5

    backward = resolve("linear()\nattention(4)", input_shape=("B", 10, 7), output_shape=("B", 10, 16))
    assert backward.nodes[0].args["out_features"] == 16
    assert backward.nodes[1].input_shapes["x"] == ("B", 10, 16)


def test_parameters_are_four_square_projections():
    plan = resolve("attention(4)", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
    assert parameter_counts(plan) == {"n0": 4 * (32 * 32 + 32)}
    model = build(plan, device="cpu")
    assert model["n0"].head_dim == 8
    assert sorted(name for name, _ in model["n0"].named_parameters()) == [
        f"{projection}_proj.{kind}" for projection in "koqv" for kind in ("bias", "weight")]


@pytest.mark.parametrize("source,shape,output_shape,message", [
    ("attention(5)", ("B", 8, 32), ("B", 8, 32), "must divide"),
    ("linear()\nattention(3)", ("B", 4, 8), ("B", 4, 16), "must divide"),
    ("attention(4, rope=True)", ("B", 8, 12), ("B", 8, 12), "even head dimension"),
])
def test_invalid_head_counts_report_a_constraint_error(source, shape, output_shape, message):
    with pytest.raises(HNDLError, match=f"E_CONSTRAINT.*{message}"):
        resolve(source, input_shape=shape, output_shape=output_shape)


def test_rank_two_input_is_rejected():
    with pytest.raises(HNDLError):
        resolve("attention(4)", input_shape=("B", 32), output_shape=("B", 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("rope", [False, True])
def test_reduced_precision_attention_runs_on_cuda(dtype, rope):
    source = f"attention(4, causal=True, rope={rope})"
    plan = resolve(source, input_shape=("B", 12, 32), output_shape=("B", 12, 32), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=2)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(3, 12, 32, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (3, 12, 32)
    assert output.isfinite().all()
    output.float().square().mean().backward()
    assert x.grad is not None and x.grad.isfinite().all()

    exact = build(resolve(source, input_shape=("B", 12, 32), output_shape=("B", 12, 32)),
                  device="cuda:0", initialization_seed=2)
    torch.testing.assert_close(output.float(), exact(x=x.detach().float())["output"], atol=3e-2, rtol=3e-2)
