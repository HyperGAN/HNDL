"""Numerics, routing semantics, inference, and error codes for the ``moe`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.moe import MixtureOfExperts
from hndl.torch import DTYPES, build


def layer(device, *, experts=4, hidden=16, top_k=2, activation="gelu", d_model=8, seed=0):
    torch.manual_seed(seed)
    module = MixtureOfExperts(experts=experts, hidden=hidden, top_k=top_k, activation=activation, D=d_model)
    return module.to(device)


def routing(module, flat):
    """Router decisions for already flattened tokens."""
    logits = module.router(flat)
    values, indices = torch.topk(logits, module.top_k, dim=-1)
    return indices, torch.softmax(values, dim=-1)


def per_token_reference(module, x):
    """Independent reference: one token at a time, one expert at a time."""
    flat = x.reshape(-1, module.d_model)
    indices, weights = routing(module, flat)
    rows = []
    for token in range(flat.shape[0]):
        total = torch.zeros_like(flat[token])
        for slot in range(module.top_k):
            total = total + weights[token, slot] * module.experts[indices[token, slot]](flat[token])
        rows.append(total)
    return torch.stack(rows).reshape(x.shape)


@pytest.mark.parametrize("shape", [(5, 8), (3, 7, 8)])
def test_forward_matches_a_per_token_dense_reference(device, shape):
    module = layer(device)
    x = torch.randn(*shape, device=device)
    torch.testing.assert_close(module(x), per_token_reference(module, x), atol=1e-5, rtol=1e-5)


def test_gate_weights_are_renormalized_over_the_selection(device):
    module = layer(device, experts=5, top_k=3)
    flat = torch.randn(16, 8, device=device)
    indices, weights = routing(module, flat)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(16, device=device))
    assert (weights > 0).all()
    assert all(len(set(row.tolist())) == 3 for row in indices)


def test_top_k_equal_to_experts_is_a_dense_softmax_mixture(device):
    module = layer(device, experts=3, top_k=3)
    x = torch.randn(6, 4, 8, device=device)
    flat = x.reshape(-1, 8)
    gates = torch.softmax(module.router(flat), dim=-1)
    dense = sum(gates[:, index, None] * expert(flat) for index, expert in enumerate(module.experts))
    torch.testing.assert_close(module(x), dense.reshape(x.shape), atol=1e-5, rtol=1e-5)


def test_top_k_one_selects_the_single_argmax_expert(device):
    module = layer(device, experts=4, top_k=1)
    x = torch.randn(9, 8, device=device)
    choice = module.router(x).argmax(dim=-1)
    expected = torch.stack([module.experts[choice[token]](x[token]) for token in range(x.shape[0])])
    torch.testing.assert_close(module(x), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("activation,reference", [
    ("gelu", lambda t: torch.nn.functional.gelu(t)),
    ("gelu_tanh", lambda t: torch.nn.functional.gelu(t, approximate="tanh")),
    ("relu", torch.relu),
    ("silu", torch.nn.functional.silu),
])
def test_every_activation_choice_matches_its_functional_form(device, activation, reference):
    module = layer(device, activation=activation)
    expert = module.experts[0]
    t = torch.randn(7, 8, device=device)
    torch.testing.assert_close(expert(t), expert.down(reference(expert.up(t))))


def test_gradients_reach_exactly_the_routed_experts(device):
    module = layer(device, experts=6, top_k=2, seed=3)
    with torch.no_grad():
        # A router that sends every strictly positive token to experts 0 and 1.
        module.router.weight.zero_()
        module.router.weight[0].fill_(10.0)
        module.router.weight[1].fill_(5.0)
    x = torch.rand(40, 8, device=device).add_(1).requires_grad_()
    selected = set(routing(module, x.detach())[0].flatten().tolist())
    assert selected == {0, 1}
    module(x).square().mean().backward()
    for index, expert in enumerate(module.experts):
        grads = [expert.up.weight.grad, expert.down.weight.grad]
        if index in selected:
            assert all(g is not None and g.abs().sum() > 0 for g in grads), f"expert {index} received no gradient"
        else:
            assert all(g is None or g.abs().sum() == 0 for g in grads), f"unrouted expert {index} received gradient"
    assert module.router.weight.grad is not None and module.router.weight.grad.isfinite().all()
    assert x.grad is not None and x.grad.isfinite().all()


def test_unrouted_tokens_do_not_leak_between_positions(device):
    module = layer(device, experts=3, top_k=1)
    x = torch.randn(4, 5, 8, device=device)
    out = module(x)
    torch.testing.assert_close(module(x[:, :1]), out[:, :1], atol=1e-5, rtol=1e-5)


def test_shapes_resolve_forward_and_backward_through_the_mixture():
    forward = resolve("moe(3, 8, top_k=1)\nlinear()", input_shape=("B", 4, 6), output_shape=("B", 4, 2))
    assert forward.nodes[0].output_shapes["out"] == ("B", 4, 6)
    assert forward.nodes[1].args["in_features"] == 6

    backward = resolve("linear()\nmoe(2, 8)\nrelu()", input_shape=("B", 7), output_shape=("B", 5))
    assert backward.nodes[0].args["out_features"] == 5
    assert backward.nodes[1].input_shapes["x"] == ("B", 5)


@pytest.mark.parametrize("source,code", [
    ("moe(2, 8, top_k=3)", "E_ARGUMENT"),
    ("moe(1, 8, top_k=2)", "E_ARGUMENT"),
    ("moe(0, 8)", "E_ARGUMENT"),
    ("moe(2, 0)", "E_ARGUMENT"),
    ('moe(2, 8, activation="tanh")', "E_ARGUMENT"),
])
def test_invalid_arguments_report_their_codes(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))


def test_image_tensors_are_rejected_with_a_constraint_error():
    with pytest.raises(HNDLError, match="E_CONSTRAINT.*flatten"):
        resolve("moe(2, 8)", input_shape=("B", 3, 4, 4), output_shape=("B", 3, 4, 4))


def test_built_plan_exposes_router_and_named_expert_parameters(device):
    plan = resolve("moe(3, 16, top_k=2)", input_shape=("B", 5, 8), output_shape=("B", 5, 8))
    model = build(plan, device=device, initialization_seed=1)
    names = dict(model["n0"].named_parameters())
    assert "router.weight" in names and "router.bias" not in names
    for index in range(3):
        for part in ("up", "down"):
            assert f"experts.{index}.{part}.weight" in names and f"experts.{index}.{part}.bias" in names
    assert sum(p.numel() for p in model["n0"].parameters()) == 3 * 8 + 3 * (16 * 8 + 16 + 8 * 16 + 8)
    out = model(x=torch.randn(2, 5, 8, device=device))["output"]
    assert out.shape == (2, 5, 8) and out.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduced precision kernels are qualified on CUDA")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_precision_routing_agrees_with_float32_on_cuda(dtype):
    plan = resolve("moe(4, 16)", input_shape=("B", 6, 8), output_shape=("B", 6, 8), dtype=dtype)
    model = build(plan, device="cuda:0", initialization_seed=7)
    module = model["n0"]
    assert all(p.dtype == DTYPES[dtype] for p in module.parameters())
    x = torch.randn(3, 6, 8, device="cuda:0", dtype=DTYPES[dtype], requires_grad=True)
    out = model(x=x)["output"]
    assert out.dtype == DTYPES[dtype] and out.shape == (3, 6, 8) and out.isfinite().all()
    out.float().square().mean().backward()
    assert x.grad.isfinite().all()
    wide = build(resolve("moe(4, 16)", input_shape=("B", 6, 8), output_shape=("B", 6, 8)),
                 device="cuda:0", initialization_seed=7)["n0"]
    torch.testing.assert_close(out.float(), wide(x.detach().float()), atol=3e-2, rtol=3e-2)
