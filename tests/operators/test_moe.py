"""Numerics, routing semantics, inference, and error codes for the ``moe`` operator."""

import pytest
import torch

from hndl import HNDLError, resolve
from hndl.operators.moe import MixtureOfExperts
from hndl.torch import DTYPES, build


def layer(device, *, experts=4, hidden=16, top_k=2, activation="gelu", expert_kind="mlp", bias=True, scoring="softmax", d_model=8, seed=0):
    torch.manual_seed(seed)
    module = MixtureOfExperts(
        experts=experts,
        hidden=hidden,
        top_k=top_k,
        activation=activation,
        expert_kind=expert_kind,
        bias=bias,
        scoring=scoring,
        D=d_model,
    )
    return module.to(device)


def routing(module, flat):
    """Router decisions for already flattened tokens."""
    logits = module.router(flat)
    values, indices = torch.topk(logits, module.top_k, dim=-1)
    if getattr(module, "scoring", "softmax") == "sigmoid":
        scores = torch.sigmoid(values)
        weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        weights = torch.softmax(values, dim=-1)
    return indices, weights


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


def test_swiglu_expert_architecture_and_numerics(device):
    module = layer(device, experts=3, hidden=16, top_k=2, expert_kind="swiglu", seed=42)
    for expert in module.experts:
        assert hasattr(expert, "gate") and hasattr(expert, "up") and hasattr(expert, "down")
        assert expert.gate.weight.shape == (16, 8)
        assert expert.up.weight.shape == (16, 8)
        assert expert.down.weight.shape == (8, 16)
    x = torch.randn(4, 5, 8, device=device)
    ref = per_token_reference(module, x)
    out = module(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_sigmoid_scoring_numerics_and_reference_parity(device):
    module = layer(device, experts=4, hidden=16, top_k=2, scoring="sigmoid", seed=99)
    x = torch.randn(3, 7, 8, device=device)
    ref = per_token_reference(module, x)
    out = module(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_bias_false_experts(device):
    module = layer(device, experts=2, hidden=12, top_k=1, bias=False, expert_kind="mlp")
    for expert in module.experts:
        assert expert.up.bias is None
        assert expert.down.bias is None
    x = torch.randn(2, 4, 8, device=device)
    torch.testing.assert_close(module(x), per_token_reference(module, x), atol=1e-5, rtol=1e-5)

    swiglu_module = layer(device, experts=2, hidden=12, top_k=1, bias=False, expert_kind="swiglu")
    for expert in swiglu_module.experts:
        assert expert.gate.bias is None
        assert expert.up.bias is None
        assert expert.down.bias is None
    torch.testing.assert_close(swiglu_module(x), per_token_reference(swiglu_module, x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("source,code", [
    ('moe(2, 8, expert_kind="invalid")', "E_ARGUMENT"),
    ('moe(2, 8, scoring="invalid")', "E_ARGUMENT"),
])
def test_invalid_sota_moe_arguments_raise_e_argument(source, code):
    with pytest.raises(HNDLError, match=code):
        resolve(source, input_shape=("B", 8), output_shape=("B", 8))


def test_moe_torch_compile_fullgraph(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    module = layer(device, experts=4, hidden=16, top_k=2, expert_kind="swiglu", scoring="sigmoid")
    x = torch.randn(4, 8, device=device)
    eager_out = module(x)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    compiled_out = compiled(x)
    torch.testing.assert_close(compiled_out, eager_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d_model,hidden", [
    (10, 16),
    (18, 18),
    (7, 13),
    (11, 17),
])
def test_unaligned_hidden_dimensions_cpu_and_cuda(device, d_model, hidden):
    """Verify that unaligned byte strides fall back cleanly to bmm on CPU and CUDA."""
    module = layer(device, experts=4, hidden=hidden, top_k=2, activation="gelu", d_model=d_model, seed=42)
    x = torch.randn(3, 5, d_model, device=device)
    ref = per_token_reference(module, x)
    out = module(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_moe_torch_compile_float16_cuda():
    """Verify that MoE compiles under torch.compile in float16 without Dynamo fake tensor crash."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    module = MixtureOfExperts(experts=4, hidden=32, top_k=2, activation="gelu", expert_kind="mlp", D=16).to(
        device="cuda:0", dtype=torch.float16
    )
    x = torch.randn(4, 16, device="cuda:0", dtype=torch.float16)
    eager_out = module(x)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    compiled_out = compiled(x)
    torch.testing.assert_close(compiled_out, eager_out, atol=1e-5, rtol=1e-5)

