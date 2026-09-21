"""Wiring, numerics, parameter names, and error codes for the ``transformer_block`` operator."""

import pytest
import torch
from torch import nn

from hndl import HNDLError, resolve
from hndl.operators.transformer_block import TransformerBlock
from hndl.torch import build, parameter_counts

from .conftest import DEVICES


def precision(device):
    """Double precision on the CPU; float32 with loose tolerances on fused CUDA kernels."""
    if device == "cpu":
        return torch.float64, {}
    return torch.float32, {"atol": 1e-5, "rtol": 1e-4}


def make_module(device, dtype, *, heads=4, width=32, mlp_ratio=4, activation="gelu", norm="layer_norm",
                causal=False, dropout=0.0, layer_scale=False, layer_scale_init=1e-5, eps=1e-5, seed=0):
    torch.manual_seed(seed)
    module = TransformerBlock(heads=heads, mlp_ratio=mlp_ratio, activation=activation, norm=norm, causal=causal,
                              dropout=dropout, layer_scale=layer_scale, layer_scale_init=layer_scale_init,
                              eps=eps, D=width)
    return module.to(device=device, dtype=dtype).eval()


def composed(module, x):
    """The pre-norm formula spelled out over the block's own submodules."""
    attended = module.attn(module.norm1(x))
    if module.gamma1 is not None:
        attended = attended * module.gamma1
    x = x + attended
    projected = module.ffn(module.norm2(x))
    if module.gamma2 is not None:
        projected = projected * module.gamma2
    return x + projected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("norm", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("layer_scale", [False, True])
@pytest.mark.parametrize("heads,width,positions", [(1, 8, 5), (4, 32, 7), (8, 16, 4), (2, 16, 1)])
def test_matches_the_composed_formula_forward_and_backward(device, norm, layer_scale, heads, width, positions):
    dtype, tolerance = precision(device)
    module = make_module(device, dtype, heads=heads, width=width, norm=norm, layer_scale=layer_scale,
                         layer_scale_init=0.1, mlp_ratio=2, seed=heads + positions)
    x = torch.randn(3, positions, width, device=device, dtype=dtype)
    actual_input = x.detach().clone().requires_grad_()
    expected_input = x.detach().clone().requires_grad_()

    actual = module(actual_input)
    expected = composed(module, expected_input)
    torch.testing.assert_close(actual, expected, **tolerance)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, **tolerance)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_matches_torch_encoder_layer_with_copied_weights(device, activation):
    """``nn.TransformerEncoderLayer(norm_first=True)`` is the same pre-norm block."""
    heads, width, mlp_ratio, positions = 4, 24, 2, 9
    module = make_module(device, torch.float32, heads=heads, width=width, mlp_ratio=mlp_ratio,
                         activation=activation, seed=11)
    torch_layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=mlp_ratio * width, dropout=0.0,
                                             activation=activation, batch_first=True, norm_first=True,
                                             layer_norm_eps=1e-5)
    torch_layer = torch_layer.to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        torch_layer.self_attn.in_proj_weight.copy_(torch.cat([module.attn.q_proj.weight, module.attn.k_proj.weight,
                                                              module.attn.v_proj.weight], dim=0))
        torch_layer.self_attn.in_proj_bias.copy_(torch.cat([module.attn.q_proj.bias, module.attn.k_proj.bias,
                                                            module.attn.v_proj.bias]))
        torch_layer.self_attn.out_proj.weight.copy_(module.attn.o_proj.weight)
        torch_layer.self_attn.out_proj.bias.copy_(module.attn.o_proj.bias)
        for ours, theirs in ((module.ffn.up, torch_layer.linear1), (module.ffn.down, torch_layer.linear2),
                             (module.norm1, torch_layer.norm1), (module.norm2, torch_layer.norm2)):
            theirs.weight.copy_(ours.weight)
            theirs.bias.copy_(ours.bias)

    x = torch.randn(2, positions, width, device=device)
    torch.testing.assert_close(module(x), torch_layer(x), atol=1e-5, rtol=1e-4)


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
def test_layer_scale_parameters_start_at_their_initial_value(device):
    module = make_module(device, torch.float32, heads=2, width=16, layer_scale=True, layer_scale_init=1e-4)
    for name in ("gamma1", "gamma2"):
        gamma = getattr(module, name)
        assert gamma.shape == (16,) and gamma.requires_grad
        torch.testing.assert_close(gamma, torch.full((16,), 1e-4, device=device))

    plain = make_module(device, torch.float32, heads=2, width=16, layer_scale=False)
    assert plain.gamma1 is None and plain.gamma2 is None
    assert "gamma1" not in plain.state_dict()


@pytest.mark.parametrize("device", DEVICES)
def test_a_tiny_layer_scale_leaves_the_block_near_the_identity(device):
    module = make_module(device, torch.float64, heads=2, width=16, layer_scale=True, layer_scale_init=1e-8, seed=4)
    x = torch.randn(2, 5, 16, device=device, dtype=torch.float64)
    torch.testing.assert_close(module(x), x, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("norm,extra", [("layer_norm", ["norm1.bias", "norm2.bias"]), ("rms_norm", [])])
def test_state_dict_keys_are_the_documented_paths(norm, extra):
    module = make_module("cpu", torch.float32, heads=2, width=8, mlp_ratio=2, norm=norm, layer_scale=True)
    expected = {"norm1.weight", "norm2.weight", "ffn.up.weight", "ffn.up.bias", "ffn.down.weight", "ffn.down.bias",
                "gamma1", "gamma2", *extra}
    expected |= {f"attn.{projection}_proj.{kind}" for projection in "qkvo" for kind in ("weight", "bias")}
    assert set(module.state_dict()) == expected
    assert module.ffn.up.weight.shape == (16, 8) and module.ffn.down.weight.shape == (8, 16)


def test_parameter_count_covers_attention_and_the_mlp():
    plan = resolve("transformer_block(4)", input_shape=("B", 8, 32), output_shape=("B", 8, 32))
    width, hidden = 32, 4 * 32
    attention = 4 * (width * width + width)
    mlp = width * hidden + hidden + hidden * width + width
    norms = 2 * 2 * width
    assert parameter_counts(plan) == {"n0": attention + mlp + norms}


def test_widths_flow_forward_and_backward_through_the_block():
    forward = resolve("transformer_block(4)\nlinear()", input_shape=("B", 8, 32), output_shape=("B", 8, 5))
    assert forward.nodes[0].output_shapes["out"] == ("B", 8, 32)
    assert forward.nodes[1].args["out_features"] == 5

    backward = resolve("linear()\ntransformer_block(4)", input_shape=("B", 10, 7), output_shape=("B", 10, 16))
    assert backward.nodes[0].args["out_features"] == 16
    assert backward.nodes[1].input_shapes["x"] == ("B", 10, 16)

    model = build(backward, device="cpu")
    assert model["n1"].ffn.up.weight.shape == (64, 16)


@pytest.mark.parametrize("source,shape,output_shape", [
    ("transformer_block(5)", ("B", 8, 32), ("B", 8, 32)),
    ("linear()\ntransformer_block(3)", ("B", 4, 8), ("B", 4, 16)),
])
def test_heads_must_divide_the_model_width(source, shape, output_shape):
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*must divide"):
        resolve(source, input_shape=shape, output_shape=output_shape)


def test_building_a_bad_head_count_directly_also_reports_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*must divide"):
        TransformerBlock(heads=3, mlp_ratio=4, activation="gelu", norm="layer_norm", causal=False, dropout=0.0,
                         layer_scale=False, layer_scale_init=1e-5, eps=1e-5, D=16)


@pytest.mark.parametrize("source", [
    'transformer_block(2, norm="batch_norm")',
    'transformer_block(2, activation="swish")',
    "transformer_block(2, mlp_ratio=0)",
    "transformer_block(2, eps=0)",
])
def test_invalid_arguments_report_an_argument_error(source):
    with pytest.raises(HNDLError, match="E_ARGUMENT"):
        resolve(source, input_shape=("B", 6, 16), output_shape=("B", 6, 16))


def test_rank_two_input_is_rejected():
    with pytest.raises(HNDLError):
        resolve("transformer_block(4)", input_shape=("B", 32), output_shape=("B", 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision is qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("norm", ["layer_norm", "rms_norm"])
def test_reduced_precision_block_runs_on_cuda(dtype, norm):
    source = f'transformer_block(4, causal=True, norm="{norm}", layer_scale=True, layer_scale_init=0.5)'
    contract = dict(input_shape=("B", 12, 32), output_shape=("B", 12, 32))
    plan = resolve(source, dtype=dtype, **contract)
    model = build(plan, device="cuda:0", initialization_seed=2)
    torch_dtype = getattr(torch, dtype)
    assert all(p.dtype == torch_dtype for p in model.parameters())
    x = torch.randn(3, 12, 32, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    output = model(x=x)["output"]
    assert output.dtype == torch_dtype and output.shape == (3, 12, 32)
    assert output.isfinite().all()
    output.float().square().mean().backward()
    assert x.grad is not None and x.grad.isfinite().all()

    exact = build(resolve(source, **contract), device="cuda:0", initialization_seed=2)
    torch.testing.assert_close(output.float(), exact(x=x.detach().float())["output"], atol=3e-2, rtol=3e-2)
