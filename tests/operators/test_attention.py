"""Numerics, position handling, and error codes for the ``attention`` operator."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from hndl import HNDLError, Registry, ResolvedPlan, resolve
from hndl.operators.attention import Attention, _reference
from hndl.torch import build, parameter_counts

from .conftest import DEVICES


def precision(device):
    """Double precision on the CPU; float32 with loose tolerances on fused CUDA kernels."""
    if device == "cpu":
        return torch.float64, {}
    return torch.float32, {"atol": 1e-5, "rtol": 1e-4}


def make_module(device, dtype, *, heads=4, width=32, causal=False, dropout=0.0, bias=True, rope=False, seed=0,
                qkv_bias=True, out_bias=True, relative_position_bias=False, spatial_shape=(0, 0), table=False):
    torch.manual_seed(seed)
    module = Attention(heads=heads, causal=causal, dropout=dropout, bias=bias, rope=rope, D=width,
                       qkv_bias=qkv_bias, out_bias=out_bias,
                       relative_position_bias=relative_position_bias, spatial_shape=spatial_shape)
    if table:
        # The table is built at zero, which would hide every indexing mistake.
        with torch.no_grad():
            module.relative_position_bias_table.normal_()
    return module.to(device=device, dtype=dtype).eval()


def hand_gathered_bias(module, dtype):
    """``[heads, T, T]`` built one (query, key) pair at a time from the raw offsets."""
    height, width = module.spatial_shape
    table = module.relative_position_bias_table
    positions = height * width

    def entry(query, key):
        dy = query // width - key // width + height - 1
        dx = query % width - key % width + width - 1
        return table[dy * (2 * width - 1) + dx]

    rows = [torch.stack([entry(query, key) for key in range(positions)]) for query in range(positions)]
    return torch.stack(rows).permute(2, 0, 1).to(dtype)


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
    if module.relative_position_bias:
        scores = scores + hand_gathered_bias(module, scores.dtype)
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


# --- Relative position bias -------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("grid,heads,width", [((2, 3), 2, 8), ((4, 4), 4, 16), ((1, 5), 1, 4), ((3, 1), 2, 4)])
def test_relative_position_bias_matches_the_explicit_oracle(device, causal, rope, grid, heads, width):
    dtype, tolerance = precision(device)
    module = make_module(device, dtype, heads=heads, width=width, causal=causal, rope=rope,
                         relative_position_bias=True, spatial_shape=grid, table=True,
                         seed=grid[0] * 10 + grid[1])
    x = torch.randn(2, grid[0] * grid[1], width, device=device, dtype=dtype)
    actual_input = x.detach().clone().requires_grad_()
    expected_input = x.detach().clone().requires_grad_()

    actual = module(actual_input)
    expected = explicit_attention(module, expected_input)
    torch.testing.assert_close(actual, expected, **tolerance)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, **tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid", [(8, 8), (16, 16), (32, 32)])
def test_global_grids_match_the_shipped_reference(device, grid):
    """The sizes HyperGAN asks for, global or used as one window of a partition."""
    positions = grid[0] * grid[1]
    module = make_module(device, torch.float32, heads=2, width=8, relative_position_bias=True,
                         spatial_shape=grid, table=True, seed=grid[0])
    x = torch.randn(1, positions, 8, device=device)
    actual, expected = module(x), _reference(module)(x)
    assert tuple(actual.shape) == (1, positions, 8)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_offset_index_lands_on_the_hand_computed_table_row():
    """Row-major tokens, table row (dy + H - 1) * (2W - 1) + (dx + W - 1)."""
    module = Attention(heads=3, causal=False, dropout=0.0, bias=True, rope=False, D=6,
                       relative_position_bias=True, spatial_shape=(2, 3))
    index = module.relative_position_index
    assert tuple(index.shape) == (6, 6)
    assert tuple(module.relative_position_bias_table.shape) == ((2 * 2 - 1) * (2 * 3 - 1), 3)

    # Token t sits at (t // 3, t % 3): 0=(0,0), 2=(0,2), 3=(1,0), 5=(1,2).
    assert int(index[0, 0]) == (0 + 1) * 5 + (0 + 2)          # a token against itself: dy=0, dx=0
    assert int(index[4, 4]) == int(index[0, 0])               # the same offset from any query
    assert int(index[0, 2]) == (0 + 1) * 5 + (-2 + 2)         # dy=0, dx=-2
    assert int(index[2, 0]) == (0 + 1) * 5 + (2 + 2)          # dy=0, dx=+2
    assert int(index[0, 3]) == (-1 + 1) * 5 + (0 + 2)         # dy=-1, dx=0
    assert int(index[5, 0]) == (1 + 1) * 5 + (2 + 2)          # dy=+1, dx=+2, the largest row
    assert int(index.max()) == (2 * 2 - 1) * (2 * 3 - 1) - 1 and int(index.min()) == 0

    # Every distinct offset occupies its own row, and equal offsets share one.
    offsets = {(query // 3 - key // 3, query % 3 - key % 3): int(index[query, key])
               for query in range(6) for key in range(6)}
    assert len(set(offsets.values())) == len(offsets)


@pytest.mark.parametrize("device", DEVICES)
def test_the_gathered_bias_shifts_the_logits_of_the_intended_pairs(device):
    """Raising one table row lifts exactly the (query, key) pairs at that offset."""
    dtype, tolerance = precision(device)
    module = make_module(device, dtype, heads=1, width=4, relative_position_bias=True, spatial_shape=(2, 3), seed=4)
    base = module.position_bias().detach()
    assert tuple(base.shape) == (1, 6, 6) and float(base.abs().max()) == 0.0
    with torch.no_grad():
        module.relative_position_bias_table[(0 + 1) * 5 + (-1 + 2)] = 1.0    # dy=0, dx=-1
    lifted = module.position_bias()
    expected = torch.tensor([[float(query // 3 == key // 3 and query % 3 - key % 3 == -1)
                              for key in range(6)] for query in range(6)], device=device, dtype=dtype)
    torch.testing.assert_close(lifted[0], expected, **tolerance)


@pytest.mark.parametrize("source,shape,message", [
    ("attention(2, spatial_shape=(4, 4), relative_position_bias=True)", ("B", 15, 8), "covers 16 tokens"),
    ("attention(2, spatial_shape=(3, 5), relative_position_bias=True)", ("B", 16, 8), "covers 15 tokens"),
    ("attention(2, relative_position_bias=True)", ("B", 16, 8), "positive H and W"),
    ("attention(2, spatial_shape=(0, 4), relative_position_bias=True)", ("B", 16, 8), "positive H and W"),
    ("attention(2, spatial_shape=(4, 4))", ("B", 16, 8), "relative_position_bias"),
    ("attention(2, spatial_shape=(-4, 4), relative_position_bias=True)", ("B", 16, 8), "must be >= 0"),
])
def test_malformed_spatial_shapes_are_rejected(source, shape, message):
    with pytest.raises(HNDLError, match=message):
        resolve(source, input_shape=shape, output_shape=shape)


def test_a_mismatched_runtime_sequence_length_is_refused():
    module = Attention(heads=2, causal=False, dropout=0.0, bias=True, rope=False, D=8,
                       relative_position_bias=True, spatial_shape=(4, 4))
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*covers 16 tokens"):
        module(torch.randn(1, 9, 8))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal", [False, True])
def test_gradients_reach_the_bias_table(device, causal):
    dtype, _ = precision(device)
    module = make_module(device, dtype, heads=2, width=8, causal=causal, relative_position_bias=True,
                         spatial_shape=(3, 4), table=True, seed=6)
    module(torch.randn(2, 12, 8, device=device, dtype=dtype)).square().mean().backward()
    table = module.relative_position_bias_table
    assert table.grad is not None and table.grad.shape == table.shape
    assert table.grad.isfinite().all() and float(table.grad.abs().max()) > 0.0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


def test_first_and_second_derivatives_of_the_bias_table_against_the_math_backend():
    torch.manual_seed(3)
    module = Attention(heads=2, causal=True, dropout=0.0, bias=True, rope=False, D=4,
                       relative_position_bias=True, spatial_shape=(2, 3)).double()
    x = torch.randn(1, 6, 4, dtype=torch.float64, requires_grad=True)
    table = module.relative_position_bias_table.detach().normal_().requires_grad_()

    def run(inputs, values):
        return torch.func.functional_call(module, {"relative_position_bias_table": values}, (inputs,))

    # The fused kernels have no double backward; the math backend is the oracle.
    with sdpa_kernel(SDPBackend.MATH):
        assert torch.autograd.gradcheck(run, (x, table))
        assert torch.autograd.gradgradcheck(run, (x, table))


@pytest.mark.parametrize("device", DEVICES)
def test_no_random_numbers_are_drawn_when_dropout_is_zero(device):
    module = make_module(device, torch.float32, heads=2, width=8, relative_position_bias=True,
                         spatial_shape=(3, 3), table=True, seed=8)
    x = torch.randn(2, 9, 8, device=device)
    for mode in (module.train, module.eval):
        mode()
        torch.manual_seed(0)
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
        module(x)
        assert torch.equal(before, torch.get_rng_state())
        if cuda_before is not None:
            assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))


def test_casting_dtype_and_device_keeps_the_bias_intact():
    module = make_module("cpu", torch.float32, heads=2, width=8, relative_position_bias=True,
                         spatial_shape=(3, 3), table=True, seed=10)
    x = torch.randn(2, 9, 8)
    expected = module(x)

    doubled = copy.deepcopy(module).to(dtype=torch.float64)
    assert doubled.relative_position_bias_table.dtype == torch.float64
    assert doubled.relative_position_index.dtype == torch.int64
    torch.testing.assert_close(doubled(x.double()).float(), expected, atol=1e-5, rtol=1e-4)

    if torch.cuda.is_available():
        moved = copy.deepcopy(module).to(device="cuda:0")
        assert moved.relative_position_index.device.type == "cuda"
        torch.testing.assert_close(moved(x.cuda()).cpu(), expected, atol=1e-5, rtol=1e-4)


def test_deepcopy_preserves_behavior_and_shares_no_state():
    module = make_module("cpu", torch.float32, heads=2, width=8, causal=True, relative_position_bias=True,
                         spatial_shape=(2, 4), table=True, seed=12)
    clone = copy.deepcopy(module)
    x = torch.randn(3, 8, 8)
    torch.testing.assert_close(clone(x), module(x))
    assert clone.relative_position_bias_table is not module.relative_position_bias_table
    assert clone.relative_position_index is not module.relative_position_index
    with torch.no_grad():
        clone.relative_position_bias_table.add_(1.0)
    assert not torch.allclose(clone(x), module(x))


def test_a_built_graph_round_trips_through_the_plan_and_a_state_dict():
    source = "attention(4, spatial_shape=(4, 4), relative_position_bias=True, qkv_bias=False, out_bias=True)"
    contract = {"input_shape": ("B", 16, 32), "output_shape": ("B", 16, 32)}
    plan = resolve(source, **contract)
    assert plan.nodes[0].args == {"heads": 4, "causal": False, "dropout": 0.0, "bias": True, "qkv_bias": False,
                                  "out_bias": True, "rope": False, "relative_position_bias": True,
                                  "spatial_shape": (4, 4)}

    restored = ResolvedPlan.from_json(plan.to_json(), registry=Registry.builtins())
    assert restored.to_json() == plan.to_json()
    assert restored.semantic_digest == plan.semantic_digest

    model = build(plan, device="cpu", initialization_seed=1)
    with torch.no_grad():
        model["n0"].relative_position_bias_table.normal_()
    x = torch.randn(2, 16, 32)
    expected = model(x=x)["output"]

    # The index is derived from spatial_shape, so it stays out of the checkpoint.
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    assert not any("relative_position_index" in name for name in state)
    assert any(name.endswith("relative_position_bias_table") for name in state)
    reloaded = build(restored, device="cpu", initialization_seed=2)
    assert not torch.allclose(reloaded(x=x)["output"], expected)
    reloaded.load_state_dict(state)
    torch.testing.assert_close(reloaded(x=x)["output"], expected)


def test_the_bias_table_is_a_named_parameter_that_settings_can_target():
    contract = {"input_shape": ("B", 6, 8), "output_shape": ("B", 6, 8)}
    plan = resolve("attention(2, spatial_shape=(2, 3), relative_position_bias=True)", **contract)
    assert parameter_counts(plan) == {"n0": 4 * (8 * 8 + 8) + (2 * 2 - 1) * (2 * 3 - 1) * 2}

    module = build(resolve('attention(2, spatial_shape=(2, 3), relative_position_bias=True,'
                           ' init={"relative_position_bias_table": 0.25},'
                           ' trainable={"relative_position_bias_table": False})', **contract), device="cpu")["n0"]
    table = module.relative_position_bias_table
    assert float(table.min()) == 0.25 and float(table.max()) == 0.25 and not table.requires_grad


# --- Independent projection biases ------------------------------------------


@pytest.mark.parametrize("bias,qkv_bias,out_bias,expected", [
    (True, True, True, "koqv"),
    (False, True, True, ""),
    (True, False, True, "o"),
    (True, True, False, "kqv"),
    (True, False, False, ""),
    (False, True, False, ""),
])
def test_qkv_bias_and_out_bias_narrow_the_shared_bias_flag(bias, qkv_bias, out_bias, expected):
    module = Attention(heads=2, causal=False, dropout=0.0, bias=bias, rope=False, D=8,
                       qkv_bias=qkv_bias, out_bias=out_bias)
    biased = "".join(sorted(letter for letter in "koqv" if getattr(module, f"{letter}_proj").bias is not None))
    assert biased == "".join(sorted(expected))
    names = sorted(name for name, _ in module.named_parameters())
    assert names == sorted([f"{letter}_proj.weight" for letter in "koqv"]
                           + [f"{letter}_proj.bias" for letter in expected])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [False, True])
def test_omitted_projection_flags_reproduce_the_shared_bias_exactly(device, bias):
    """``bias=`` on its own must behave as it did before the narrow flags existed."""
    dtype, tolerance = precision(device)
    shared = make_module(device, dtype, heads=2, width=8, bias=bias, seed=13)
    narrow = make_module(device, dtype, heads=2, width=8, bias=bias, qkv_bias=True, out_bias=True, seed=13)
    x = torch.randn(2, 5, 8, device=device, dtype=dtype)
    torch.testing.assert_close(narrow(x), shared(x), **tolerance)
    assert narrow.state_dict().keys() == shared.state_dict().keys()


def test_the_transgan_generator_call_resolves_and_runs():
    source = "attention(4, spatial_shape=(16, 16), relative_position_bias=True, qkv_bias=False, out_bias=True)"
    plan = resolve(source, input_shape=("B", 256, 32), output_shape=("B", 256, 32))
    assert parameter_counts(plan) == {"n0": 3 * 32 * 32 + (32 * 32 + 32) + (2 * 16 - 1) ** 2 * 4}
    model = build(plan, device="cpu", initialization_seed=0)
    output = model(x=torch.randn(2, 256, 32))["output"]
    assert tuple(output.shape) == (2, 256, 32) and output.isfinite().all()
